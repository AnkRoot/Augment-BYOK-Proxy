use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use anyhow::Context;
use reqwest::header::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::anthropic::{AnthropicRequest, AnthropicResponse};
use crate::config::{
  AbridgedHistoryParams, AnthropicProviderConfig, Config, OpenAICompatibleProviderConfig,
  ProviderConfig,
};
use crate::convert::{convert_augment_to_anthropic, convert_augment_to_openai_compatible};
use crate::history_summary::compact_chat_history;
use crate::openai::OpenAIChatCompletionRequest;
use crate::protocol::{
  has_history_summary_node, AugmentChatHistory, AugmentRequest, NodeIn, REQUEST_NODE_FILE,
  REQUEST_NODE_FILE_ID, REQUEST_NODE_HISTORY_SUMMARY, REQUEST_NODE_IMAGE, REQUEST_NODE_IMAGE_ID,
  REQUEST_NODE_TOOL_RESULT, RESPONSE_NODE_MAIN_TEXT_FINISHED, RESPONSE_NODE_RAW_RESPONSE,
  RESPONSE_NODE_TOOL_USE, RESPONSE_NODE_TOOL_USE_START,
};
use crate::util::{join_url, normalize_raw_token, now_ms};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct HistorySummaryCache {
  #[serde(default)]
  entries: HashMap<String, RollingSummaryState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RollingSummaryState {
  summary_text: String,
  summarized_until_request_id: String,
  summarization_request_id: String,
  updated_at_ms: u64,
}

impl HistorySummaryCache {
  pub async fn load_from_file(path: &Path) -> anyhow::Result<Self> {
    let bytes = match tokio::fs::read(path).await {
      Ok(v) => v,
      Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
      Err(err) => {
        return Err(err)
          .with_context(|| format!("读取 history_summary cache 失败: {}", path.display()))
      }
    };

    #[derive(Debug, Deserialize)]
    struct CacheFile {
      #[serde(default)]
      entries: HashMap<String, RollingSummaryState>,
    }

    let parsed: CacheFile =
      serde_json::from_slice(&bytes).context("解析 history_summary cache JSON 失败")?;
    Ok(Self {
      entries: parsed.entries,
    })
  }

  pub async fn save_to_file(&self, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
      tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("创建 cache 目录失败: {}", parent.display()))?;
    }

    #[derive(Debug, Serialize)]
    struct CacheFile<'a> {
      version: u32,
      entries: &'a HashMap<String, RollingSummaryState>,
    }

    let json = serde_json::to_vec_pretty(&CacheFile {
      version: 1,
      entries: &self.entries,
    })
    .context("序列化 history_summary cache JSON 失败")?;

    let tmp = path.with_extension("json.tmp");
    tokio::fs::write(&tmp, json)
      .await
      .with_context(|| format!("写入临时 cache 文件失败: {}", tmp.display()))?;
    if let Err(err) = tokio::fs::rename(&tmp, path).await {
      let _ = tokio::fs::remove_file(path).await;
      tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("覆盖写入 cache 文件失败: {} ({err})", path.display()))?;
    }
    Ok(())
  }

  pub fn remove_conversation(&mut self, conversation_id: &str) -> bool {
    let key = conversation_id.trim();
    if key.is_empty() {
      return false;
    }
    self.entries.remove(key).is_some()
  }

  pub fn clear_all(&mut self) {
    self.entries.clear();
  }

  fn get_fresh(
    &self,
    conversation_id: &str,
    boundary_request_id: &str,
    now_ms: u64,
    ttl_ms: u64,
  ) -> Option<(String, String)> {
    let state = self.get_fresh_state(conversation_id, now_ms, ttl_ms)?;
    if state.summarized_until_request_id != boundary_request_id {
      return None;
    }
    Some((state.summary_text, state.summarization_request_id))
  }

  fn get_fresh_state(
    &self,
    conversation_id: &str,
    now_ms: u64,
    ttl_ms: u64,
  ) -> Option<RollingSummaryState> {
    let entry = self.entries.get(conversation_id)?;
    if ttl_ms > 0 && now_ms.saturating_sub(entry.updated_at_ms) > ttl_ms {
      return None;
    }
    Some(entry.clone())
  }

  fn put(
    &mut self,
    conversation_id: &str,
    boundary_request_id: &str,
    summary_text: String,
    summarization_request_id: String,
    now_ms: u64,
  ) {
    self.put_state(
      conversation_id,
      RollingSummaryState {
        summary_text,
        summarized_until_request_id: boundary_request_id.to_string(),
        summarization_request_id,
        updated_at_ms: now_ms,
      },
    );
  }

  fn put_state(&mut self, conversation_id: &str, state: RollingSummaryState) {
    self.entries.insert(conversation_id.to_string(), state);
  }
}

fn approx_token_count_from_byte_len(len: usize) -> u32 {
  const BYTES_PER_TOKEN: usize = 4;
  let tokens = len
    .saturating_add(BYTES_PER_TOKEN.saturating_sub(1))
    .saturating_div(BYTES_PER_TOKEN);
  u32::try_from(tokens).unwrap_or(u32::MAX)
}

fn resolve_context_window_tokens(
  hs: &crate::config::HistorySummaryConfig,
  requested_model: &str,
) -> Option<u32> {
  let model = requested_model.trim();
  if model.is_empty() {
    return None;
  }

  if !hs.context_window_tokens_overrides.is_empty() {
    let mut entries: Vec<(&String, &u32)> = hs.context_window_tokens_overrides.iter().collect();
    entries.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    for (k, v) in entries {
      let k = k.trim();
      if k.is_empty() {
        continue;
      }
      if model.contains(k) && *v > 0 {
        return Some(*v);
      }
    }
  }

  if hs.context_window_tokens_default > 0 {
    Some(hs.context_window_tokens_default)
  } else {
    infer_context_window_tokens_from_model_name(model)
  }
}

fn infer_context_window_tokens_from_model_name(model: &str) -> Option<u32> {
  let m = model.trim().to_ascii_lowercase();
  if m.is_empty() {
    return None;
  }

  if m.contains("gemini-2.5-pro") {
    return Some(1_000_000);
  }
  if m.contains("claude-") {
    return Some(200_000);
  }
  if m.contains("gpt-4o") {
    return Some(128_000);
  }

  let b = m.as_bytes();
  let mut i = 0usize;
  while i < b.len() {
    if !b[i].is_ascii_digit() {
      i += 1;
      continue;
    }
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
      i += 1;
    }
    let end = i;
    let digits_len = end.saturating_sub(start);
    if digits_len == 0 || digits_len > 4 {
      continue;
    }
    if i >= b.len() || b[i] != b'k' {
      continue;
    }

    let has_left_boundary = start == 0 || !b[start - 1].is_ascii_digit();
    let has_right_boundary = i + 1 == b.len() || !b[i + 1].is_ascii_digit();
    if !has_left_boundary || !has_right_boundary {
      continue;
    }

    let Ok(n) = m[start..end].parse::<u32>() else {
      continue;
    };
    if n == 0 {
      continue;
    }
    if n == 128 {
      return Some(128_000);
    }
    if n == 200 {
      return Some(200_000);
    }
    return Some(n.saturating_mul(1024));
  }
  None
}

#[derive(Debug, Clone, Copy)]
enum TriggerDecision {
  NotTriggered,
  TriggerChars {
    threshold_chars: usize,
  },
  TriggerRatio {
    context_window_tokens: u32,
    threshold_chars: usize,
    target_tail_budget_chars: usize,
    approx_total_tokens: u32,
    approx_ratio: f32,
  },
}

fn node_is_tool_result(n: &NodeIn) -> bool {
  n.node_type == REQUEST_NODE_TOOL_RESULT && n.tool_result_node.is_some()
}

fn history_contains_summary(history: &[AugmentChatHistory]) -> bool {
  history.iter().any(|h| {
    has_history_summary_node(&h.request_nodes)
      || has_history_summary_node(&h.structured_request_nodes)
      || has_history_summary_node(&h.nodes)
  })
}

fn exchange_request_nodes<'a>(h: &'a AugmentChatHistory) -> impl Iterator<Item = &'a NodeIn> + 'a {
  h.request_nodes
    .iter()
    .chain(&h.structured_request_nodes)
    .chain(&h.nodes)
}

fn exchange_response_nodes<'a>(h: &'a AugmentChatHistory) -> impl Iterator<Item = &'a NodeIn> + 'a {
  h.response_nodes.iter().chain(&h.structured_output_nodes)
}

fn exchange_has_tool_results(h: &AugmentChatHistory) -> bool {
  exchange_request_nodes(h).any(node_is_tool_result)
}

fn estimate_node_size_chars(node: &NodeIn) -> usize {
  let mut n = 16usize;
  n += node.content.len();
  if let Some(t) = node.text_node.as_ref() {
    n += t.content.len();
  }
  if let Some(tr) = node.tool_result_node.as_ref() {
    n += tr.tool_use_id.len();
    n += tr.content.len();
    for c in &tr.content_nodes {
      n += 8;
      n += c.text_content.len();
      if let Some(img) = c.image_content.as_ref() {
        n += img.image_data.len();
      }
    }
  }
  if let Some(img) = node.image_node.as_ref() {
    n += img.image_data.len();
  }
  for v in [
    node.image_id_node.as_ref(),
    node.ide_state_node.as_ref(),
    node.edit_events_node.as_ref(),
    node.checkpoint_ref_node.as_ref(),
    node.change_personality_node.as_ref(),
    node.file_node.as_ref(),
    node.file_id_node.as_ref(),
    node.history_summary_node.as_ref(),
  ]
  .into_iter()
  .flatten()
  {
    n += v.to_string().len();
  }
  if let Some(tu) = node.tool_use.as_ref() {
    n += tu.tool_use_id.len();
    n += tu.tool_name.len();
    n += tu.input_json.len();
    n += tu.mcp_server_name.len();
    n += tu.mcp_tool_name.len();
  }
  if let Some(th) = node.thinking.as_ref() {
    n += th.summary.len();
  }
  n
}

fn estimate_exchange_size_chars(h: &AugmentChatHistory) -> usize {
  let mut n = 0usize;

  let req_nodes: Vec<&NodeIn> = exchange_request_nodes(h).collect();
  if !req_nodes.is_empty() {
    n += req_nodes
      .iter()
      .map(|x| estimate_node_size_chars(x))
      .sum::<usize>();
  } else {
    n += h.request_message.len();
  }

  let resp_nodes: Vec<&NodeIn> = exchange_response_nodes(h).collect();
  if !resp_nodes.is_empty() {
    n += resp_nodes
      .iter()
      .map(|x| estimate_node_size_chars(x))
      .sum::<usize>();
  } else {
    n += h.response_text.len();
  }

  n
}

fn estimate_history_size_chars(history: &[AugmentChatHistory]) -> usize {
  history.iter().map(estimate_exchange_size_chars).sum()
}

fn estimate_request_extra_size_chars(augment: &AugmentRequest) -> usize {
  let ctx = augment.context.as_ref();

  let prefix = if !augment.prefix.trim().is_empty() {
    augment.prefix.as_str()
  } else {
    ctx.map(|c| c.prefix.as_str()).unwrap_or("")
  };
  let selected_code = if !augment.selected_code.trim().is_empty() {
    augment.selected_code.as_str()
  } else {
    ctx.map(|c| c.selected_code.as_str()).unwrap_or("")
  };
  let suffix = if !augment.suffix.trim().is_empty() {
    augment.suffix.as_str()
  } else {
    ctx.map(|c| c.suffix.as_str()).unwrap_or("")
  };
  let diff = if !augment.diff.trim().is_empty() {
    augment.diff.as_str()
  } else {
    ctx.map(|c| c.diff.as_str()).unwrap_or("")
  };

  prefix.len() + selected_code.len() + suffix.len() + diff.len()
}

#[derive(Debug, Clone)]
struct HistorySplit {
  head: Vec<AugmentChatHistory>,
  tail: Vec<AugmentChatHistory>,
}

fn split_history_for_summary(
  history: &[AugmentChatHistory],
  tail_size_chars_to_exclude: usize,
  trigger_on_history_size_chars: usize,
  min_tail_exchanges: usize,
) -> HistorySplit {
  if history.is_empty() {
    return HistorySplit {
      head: Vec::new(),
      tail: Vec::new(),
    };
  }

  let mut head_rev: Vec<AugmentChatHistory> = Vec::new();
  let mut tail_rev: Vec<AugmentChatHistory> = Vec::new();
  let mut seen_chars = 0usize;
  let mut head_chars = 0usize;
  let mut tail_chars = 0usize;

  for h in history.iter().rev() {
    let sz = estimate_exchange_size_chars(h);
    if seen_chars.saturating_add(sz) < tail_size_chars_to_exclude
      || tail_rev.len() < min_tail_exchanges
    {
      tail_rev.push(h.clone());
      tail_chars = tail_chars.saturating_add(sz);
    } else {
      head_rev.push(h.clone());
      head_chars = head_chars.saturating_add(sz);
    }
    seen_chars = seen_chars.saturating_add(sz);
  }

  let total_chars = head_chars.saturating_add(tail_chars);
  if total_chars < trigger_on_history_size_chars {
    let mut all_rev = tail_rev;
    all_rev.extend(head_rev);
    all_rev.reverse();
    return HistorySplit {
      head: Vec::new(),
      tail: all_rev,
    };
  }

  head_rev.reverse();
  tail_rev.reverse();
  HistorySplit {
    head: head_rev,
    tail: tail_rev,
  }
}

fn adjust_tail_to_avoid_tool_result_orphans(
  original: &[AugmentChatHistory],
  mut tail_start: usize,
) -> usize {
  while tail_start < original.len() {
    let has_tr = exchange_has_tool_results(&original[tail_start]);
    if !has_tr {
      break;
    }
    if tail_start == 0 {
      break;
    }
    tail_start -= 1;
  }
  tail_start
}

#[derive(Debug, Clone)]
struct AgentActionsSummary {
  files_modified: HashSet<String>,
  files_created: HashSet<String>,
  files_deleted: HashSet<String>,
  files_viewed: HashSet<String>,
  terminal_commands: HashSet<String>,
}

impl Default for AgentActionsSummary {
  fn default() -> Self {
    Self {
      files_modified: HashSet::new(),
      files_created: HashSet::new(),
      files_deleted: HashSet::new(),
      files_viewed: HashSet::new(),
      terminal_commands: HashSet::new(),
    }
  }
}

#[derive(Debug, Clone)]
struct AbridgedEntry {
  user_message: String,
  agent_actions_summary: AgentActionsSummary,
  agent_final_response: String,
  was_interrupted: bool,
  continues: bool,
}

fn node_has_image_or_file_marker(n: &NodeIn) -> (bool, bool) {
  let has_image = matches!(n.node_type, REQUEST_NODE_IMAGE | REQUEST_NODE_IMAGE_ID)
    || n.image_node.is_some()
    || n.image_id_node.is_some();
  let has_file = matches!(n.node_type, REQUEST_NODE_FILE | REQUEST_NODE_FILE_ID)
    || n.file_node.is_some()
    || n.file_id_node.is_some();
  (has_image, has_file)
}

fn build_user_message_with_attachments(h: &AugmentChatHistory) -> String {
  let mut msg = h.request_message.clone();
  let mut has_image = false;
  let mut has_file = false;
  for n in exchange_request_nodes(h) {
    let (i, f) = node_has_image_or_file_marker(n);
    has_image |= i;
    has_file |= f;
    if has_image && has_file {
      break;
    }
  }
  if has_image {
    msg.push_str("\n[User attached image]");
  }
  if has_file {
    msg.push_str("\n[User attached document]");
  }
  msg
}

fn iter_response_tool_uses<'a>(
  h: &'a AugmentChatHistory,
) -> impl Iterator<Item = &'a crate::protocol::ToolUse> + 'a {
  exchange_response_nodes(h).filter_map(|n| {
    if matches!(
      n.node_type,
      RESPONSE_NODE_TOOL_USE | RESPONSE_NODE_TOOL_USE_START
    ) {
      n.tool_use.as_ref()
    } else {
      None
    }
  })
}

fn extract_assistant_text_from_output_nodes<'a>(nodes: impl Iterator<Item = &'a NodeIn>) -> String {
  let mut finished = String::new();
  let mut raw = String::new();
  for n in nodes {
    if n.node_type == RESPONSE_NODE_MAIN_TEXT_FINISHED && !n.content.trim().is_empty() {
      finished = n.content.clone();
    } else if n.node_type == RESPONSE_NODE_RAW_RESPONSE && !n.content.trim().is_empty() {
      raw.push_str(&n.content);
    }
  }
  let out = if !finished.trim().is_empty() {
    finished
  } else {
    raw
  };
  out.trim().to_string()
}

fn add_tool_use_to_actions(tool_use: &crate::protocol::ToolUse, actions: &mut AgentActionsSummary) {
  let Ok(v) = serde_json::from_str::<Value>(tool_use.input_json.as_str()) else {
    return;
  };
  match tool_use.tool_name.as_str() {
    "str-replace-editor" => {
      if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
        if !p.trim().is_empty() {
          actions.files_modified.insert(p.trim().to_string());
        }
      }
    }
    "save-file" => {
      if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
        if !p.trim().is_empty() {
          actions.files_created.insert(p.trim().to_string());
        }
      }
    }
    "remove-files" => {
      if let Some(arr) = v.get("file_paths").and_then(|x| x.as_array()) {
        for p in arr.iter().filter_map(|x| x.as_str()) {
          if !p.trim().is_empty() {
            actions.files_deleted.insert(p.trim().to_string());
          }
        }
      }
    }
    "view" => {
      if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
        if !p.trim().is_empty() {
          actions.files_viewed.insert(p.trim().to_string());
        }
      }
    }
    "launch-process" => {
      if let Some(c) = v.get("command").and_then(|x| x.as_str()) {
        if !c.trim().is_empty() {
          actions.terminal_commands.insert(c.trim().to_string());
        }
      }
    }
    _ => {}
  }
}

fn finalize_actions(actions: &mut AgentActionsSummary) {
  for p in actions.files_modified.clone() {
    actions.files_viewed.remove(p.as_str());
  }
}

fn middle_truncate_with_ellipsis(
  s: &str,
  limit: usize,
  start_ratio: f32,
  end_ratio: f32,
) -> String {
  if limit == 0 {
    return String::new();
  }
  let chars: Vec<char> = s.chars().collect();
  if chars.len() <= limit {
    return s.to_string();
  }
  if start_ratio + end_ratio > 1.0 {
    return chars.into_iter().take(limit).collect();
  }
  if limit <= 3 {
    return "...".chars().take(limit).collect();
  }
  let keep = limit - 3;
  let start = ((keep as f32) * start_ratio).floor() as usize;
  let end = ((keep as f32) * end_ratio).floor() as usize;
  let start = start.min(keep);
  let end = end.min(keep.saturating_sub(start));
  let mut out = String::new();
  out.extend(chars.iter().take(start));
  out.push_str("...");
  out.extend(chars.iter().skip(chars.len().saturating_sub(end)));
  out
}

fn limit_set_items(
  set: &HashSet<String>,
  max_items: usize,
  item_char_limit: usize,
  noun: &str,
) -> Vec<String> {
  if set.is_empty() {
    return Vec::new();
  }
  let mut items: Vec<String> = set.iter().cloned().collect();
  items.sort();
  let trunc = |s: String| middle_truncate_with_ellipsis(&s, item_char_limit, 0.5, 0.5);
  if items.len() <= max_items {
    return items.into_iter().map(trunc).collect();
  }
  let mut out: Vec<String> = items.into_iter().take(max_items).map(trunc).collect();
  let remaining = set.len().saturating_sub(max_items);
  out.push(format!("... {remaining} more {noun}"));
  out
}

fn render_abridged_entry(entry: &AbridgedEntry, params: &AbridgedHistoryParams) -> String {
  let mut user_message = entry.user_message.clone();
  if user_message.chars().count() > params.user_message_chars_limit {
    user_message = middle_truncate_with_ellipsis(
      user_message.as_str(),
      params.user_message_chars_limit,
      0.5,
      0.5,
    );
  }

  let mut agent_response = entry.agent_final_response.clone();
  if agent_response.chars().count() > params.agent_response_chars_limit {
    agent_response = middle_truncate_with_ellipsis(
      agent_response.as_str(),
      params.agent_response_chars_limit,
      0.5,
      0.5,
    );
  }

  let has_actions = !entry.agent_actions_summary.files_modified.is_empty()
    || !entry.agent_actions_summary.files_created.is_empty()
    || !entry.agent_actions_summary.files_deleted.is_empty()
    || !entry.agent_actions_summary.files_viewed.is_empty()
    || !entry.agent_actions_summary.terminal_commands.is_empty();

  let mut out = String::new();
  if !user_message.trim().is_empty() {
    out.push_str("<user_request>\n");
    out.push_str(user_message.trim_end_matches('\n'));
    out.push_str("\n</user_request>\n");
  }

  if has_actions {
    out.push_str("<agent_actions_summary>\n");
    let files_modified = limit_set_items(
      &entry.agent_actions_summary.files_modified,
      params.num_files_modified_limit,
      params.action_chars_limit,
      "files",
    );
    if !files_modified.is_empty() {
      out.push_str("<files_modified>\n");
      for p in files_modified {
        out.push_str(p.trim_end_matches('\n'));
        out.push('\n');
      }
      out.push_str("</files_modified>\n");
    }
    let files_created = limit_set_items(
      &entry.agent_actions_summary.files_created,
      params.num_files_created_limit,
      params.action_chars_limit,
      "files",
    );
    if !files_created.is_empty() {
      out.push_str("<files_created>\n");
      for p in files_created {
        out.push_str(p.trim_end_matches('\n'));
        out.push('\n');
      }
      out.push_str("</files_created>\n");
    }
    let files_deleted = limit_set_items(
      &entry.agent_actions_summary.files_deleted,
      params.num_files_deleted_limit,
      params.action_chars_limit,
      "files",
    );
    if !files_deleted.is_empty() {
      out.push_str("<files_deleted>\n");
      for p in files_deleted {
        out.push_str(p.trim_end_matches('\n'));
        out.push('\n');
      }
      out.push_str("</files_deleted>\n");
    }
    let files_viewed = limit_set_items(
      &entry.agent_actions_summary.files_viewed,
      params.num_files_viewed_limit,
      params.action_chars_limit,
      "files",
    );
    if !files_viewed.is_empty() {
      out.push_str("<files_viewed>\n");
      for p in files_viewed {
        out.push_str(p.trim_end_matches('\n'));
        out.push('\n');
      }
      out.push_str("</files_viewed>\n");
    }
    let terminal_commands = limit_set_items(
      &entry.agent_actions_summary.terminal_commands,
      params.num_terminal_commands_limit,
      params.action_chars_limit,
      "commands",
    );
    if !terminal_commands.is_empty() {
      out.push_str("<terminal_commands>\n");
      for c in terminal_commands {
        out.push_str(c.trim_end_matches('\n'));
        out.push('\n');
      }
      out.push_str("</terminal_commands>\n");
    }
    out.push_str("</agent_actions_summary>\n");
  }

  if !agent_response.trim().is_empty() {
    out.push_str("<agent_response>\n");
    out.push_str(agent_response.trim_end_matches('\n'));
    out.push_str("\n</agent_response>\n");
  } else if entry.was_interrupted {
    out.push_str("<agent_was_interrupted/>\n");
  } else if entry.continues {
    out.push_str("<agent_continues/>\n");
  }

  out.trim().to_string()
}

fn build_abridged_entries(history: &[AugmentChatHistory]) -> Vec<AbridgedEntry> {
  let mut out: Vec<AbridgedEntry> = Vec::new();
  let mut current: Option<AbridgedEntry> = None;

  for h in history {
    let is_tool_result_exchange =
      exchange_request_nodes(h).any(|n| n.node_type == REQUEST_NODE_TOOL_RESULT);
    if !is_tool_result_exchange {
      if let Some(mut prev) = current.take() {
        if prev.agent_final_response.trim().is_empty() {
          prev.was_interrupted = true;
        }
        out.push(prev);
      }
      current = Some(AbridgedEntry {
        user_message: build_user_message_with_attachments(h),
        agent_actions_summary: AgentActionsSummary::default(),
        agent_final_response: String::new(),
        was_interrupted: false,
        continues: false,
      });
    }

    let Some(cur) = current.as_mut() else {
      continue;
    };

    let mut saw_tool_use = false;
    for tu in iter_response_tool_uses(h) {
      saw_tool_use = true;
      add_tool_use_to_actions(tu, &mut cur.agent_actions_summary);
    }
    if !saw_tool_use {
      let resp = if !h.response_text.trim().is_empty() {
        h.response_text.clone()
      } else {
        extract_assistant_text_from_output_nodes(exchange_response_nodes(h))
      };
      if !resp.trim().is_empty() {
        cur.agent_final_response = resp;
      }
    }
  }

  if let Some(mut last) = current.take() {
    if last.agent_final_response.trim().is_empty() {
      last.continues = true;
    }
    out.push(last);
  }

  for e in &mut out {
    finalize_actions(&mut e.agent_actions_summary);
  }

  out
}

fn build_abridged_history_text(
  history: &[AugmentChatHistory],
  params: &AbridgedHistoryParams,
  until_request_id: Option<&str>,
) -> (String, usize) {
  let mut slice = history;
  if let Some(rid) = until_request_id.filter(|s| !s.trim().is_empty()) {
    if let Some(pos) = history.iter().position(|h| h.request_id == rid) {
      slice = &history[..pos];
    }
  }

  let entries = build_abridged_entries(slice);
  let mut total = 0usize;
  let mut rendered_rev: Vec<String> = Vec::new();
  let mut dropped_beginning = 0usize;

  for (idx_from_end, e) in entries.iter().rev().enumerate() {
    let rendered = render_abridged_entry(e, params);
    if total.saturating_add(rendered.len()) > params.total_chars_limit {
      dropped_beginning = entries.len().saturating_sub(idx_from_end);
      break;
    }
    total = total.saturating_add(rendered.len());
    rendered_rev.push(rendered);
  }

  rendered_rev.reverse();
  let text = rendered_rev.join("\n").trim().to_string();
  (text, dropped_beginning)
}

#[derive(Debug, Clone, Copy)]
enum SummaryProviderRef<'a> {
  Anthropic(&'a AnthropicProviderConfig),
  OpenAICompatible(&'a OpenAICompatibleProviderConfig),
}

fn get_byok_provider_by_id<'a>(
  cfg: &'a Config,
  provider_id: &str,
) -> anyhow::Result<SummaryProviderRef<'a>> {
  let pid = provider_id.trim();
  if pid.is_empty() {
    anyhow::bail!("provider_id 为空");
  }
  for p in &cfg.byok.providers {
    match p {
      ProviderConfig::Anthropic(p) if p.id.trim() == pid => {
        return Ok(SummaryProviderRef::Anthropic(p))
      }
      ProviderConfig::OpenAICompatible(p) if p.id.trim() == pid => {
        return Ok(SummaryProviderRef::OpenAICompatible(p))
      }
      _ => {}
    }
  }
  Err(anyhow::anyhow!("未找到 provider: {pid}"))
}

fn extract_openai_choice_text(v: &Value) -> String {
  let Some(choices) = v.get("choices").and_then(|x| x.as_array()) else {
    return String::new();
  };
  let Some(first) = choices.first() else {
    return String::new();
  };
  let msg = first.get("message").or_else(|| first.get("delta"));
  let Some(msg) = msg else {
    return String::new();
  };
  let c = msg.get("content");
  match c {
    Some(Value::String(s)) => s.clone(),
    Some(Value::Array(items)) => items
      .iter()
      .filter_map(|it| it.get("text").and_then(|x| x.as_str()))
      .collect::<Vec<_>>()
      .join(""),
    _ => String::new(),
  }
}

fn extract_anthropic_text(resp: &AnthropicResponse) -> String {
  let mut out = String::new();
  for b in &resp.content {
    if let Some(t) = b.text.as_deref() {
      if !t.is_empty() {
        out.push_str(t);
      }
    }
  }
  out
}

async fn run_summary_model_once(
  http: &reqwest::Client,
  provider: SummaryProviderRef<'_>,
  prompt: &str,
  chat_history: Vec<AugmentChatHistory>,
  max_tokens: u32,
  timeout_seconds: u64,
  model: String,
) -> anyhow::Result<(String, String)> {
  let augment = AugmentRequest {
    model: None,
    chat_history,
    message: prompt.to_string(),
    agent_memories: String::new(),
    mode: String::new(),
    prefix: String::new(),
    selected_code: String::new(),
    suffix: String::new(),
    diff: String::new(),
    lang: String::new(),
    path: String::new(),
    user_guidelines: String::new(),
    workspace_guidelines: String::new(),
    rules: Value::Null,
    tool_definitions: Vec::new(),
    nodes: Vec::new(),
    structured_request_nodes: Vec::new(),
    request_nodes: Vec::new(),
    conversation_id: None,
    context: None,
  };

  match provider {
    SummaryProviderRef::Anthropic(p) => {
      let url = join_url(&p.base_url, "messages").context("anthropic base_url 无效")?;
      let api_key = normalize_raw_token(&p.api_key);
      if api_key.is_empty() {
        anyhow::bail!("history_summary provider({}) api_key 为空", p.id);
      }

      let mut req: AnthropicRequest = convert_augment_to_anthropic(p, &augment, model)?;
      req.stream = false;
      req.max_tokens = max_tokens;
      req.tools = None;
      req.tool_choice = None;
      req.thinking = None;

      let mut r = http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("x-api-key", api_key)
        .timeout(Duration::from_secs(timeout_seconds))
        .json(&req);

      for (k, v) in &p.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          r = r.header(k, value);
        }
      }

      let resp = r.send().await.context("上游请求失败")?;
      if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("上游返回错误: {status} {body}");
      }
      let body: AnthropicResponse = resp.json().await.context("解析 Anthropic 响应失败")?;
      Ok((body.id.clone(), extract_anthropic_text(&body)))
    }
    SummaryProviderRef::OpenAICompatible(p) => {
      let url = join_url(&p.base_url, "chat/completions").context("openai base_url 无效")?;
      let api_key = normalize_raw_token(&p.api_key);
      if api_key.is_empty() {
        anyhow::bail!("history_summary provider({}) api_key 为空", p.id);
      }

      let mut req: OpenAIChatCompletionRequest =
        convert_augment_to_openai_compatible(p, &augment, model)?;
      req.stream = false;
      req.stream_options = None;
      req.max_tokens = Some(max_tokens);
      req.tools = None;
      req.tool_choice = None;

      let mut r = http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .header("authorization", format!("Bearer {api_key}"))
        .timeout(Duration::from_secs(timeout_seconds))
        .json(&req);

      for (k, v) in &p.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          r = r.header(k, value);
        }
      }

      let resp = r.send().await.context("上游请求失败")?;
      if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("上游返回错误: {status} {body}");
      }
      let body: Value = resp.json().await.context("解析 OpenAI 响应失败")?;
      let id = body
        .get("id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
      let text = extract_openai_choice_text(&body);
      Ok((id, text))
    }
  }
}

fn build_history_summary_node(id: i32, history_summary_node: Value) -> NodeIn {
  NodeIn {
    id,
    node_type: REQUEST_NODE_HISTORY_SUMMARY,
    content: String::new(),
    text_node: None,
    tool_result_node: None,
    image_node: None,
    image_id_node: None,
    ide_state_node: None,
    edit_events_node: None,
    checkpoint_ref_node: None,
    change_personality_node: None,
    file_node: None,
    file_id_node: None,
    history_summary_node: Some(history_summary_node),
    tool_use: None,
    thinking: None,
  }
}

fn build_history_end_exchanges(tail: &[AugmentChatHistory]) -> Vec<Value> {
  tail
    .iter()
    .map(|h| {
      let req_nodes: Vec<NodeIn> = exchange_request_nodes(h).cloned().collect();
      let resp_nodes: Vec<NodeIn> = exchange_response_nodes(h).cloned().collect();
      serde_json::json!({
        "request_id": h.request_id,
        "request_message": h.request_message,
        "response_text": h.response_text,
        "request_nodes": req_nodes,
        "response_nodes": resp_nodes,
      })
    })
    .collect()
}

fn take_tool_result_nodes(nodes: &mut Vec<NodeIn>) -> Vec<NodeIn> {
  let mut out = Vec::new();
  let mut i = 0usize;
  while i < nodes.len() {
    if nodes[i].node_type == REQUEST_NODE_TOOL_RESULT && nodes[i].tool_result_node.is_some() {
      out.push(nodes.remove(i));
      continue;
    }
    i += 1;
  }
  out
}

pub async fn maybe_summarize_and_compact(
  http: &reqwest::Client,
  cfg: &Config,
  cache: &RwLock<HistorySummaryCache>,
  cache_path: &Path,
  chat_provider_id: &str,
  chat_model: &str,
  augment: &mut AugmentRequest,
) -> anyhow::Result<bool> {
  let hs = &cfg.history_summary;
  if !hs.enabled {
    return Ok(false);
  }

  let conv_id = augment
    .conversation_id
    .as_deref()
    .map(str::trim)
    .filter(|s| !s.is_empty())
    .unwrap_or("");
  if conv_id.is_empty() {
    return Ok(false);
  }

  if augment.chat_history.is_empty() {
    return Ok(false);
  }

  if history_contains_summary(&augment.chat_history) {
    return Ok(false);
  }

  let total_chars = estimate_history_size_chars(&augment.chat_history);
  let total_with_extra = total_chars
    .saturating_add(augment.message.len())
    .saturating_add(estimate_request_extra_size_chars(augment));

  let strategy = hs.trigger_strategy.trim().to_ascii_lowercase();
  let cw_tokens_raw = resolve_context_window_tokens(hs, chat_model);

  let decision = match strategy.as_str() {
    "chars" => {
      if total_with_extra < hs.trigger_on_history_size_chars {
        TriggerDecision::NotTriggered
      } else {
        TriggerDecision::TriggerChars {
          threshold_chars: hs.trigger_on_history_size_chars,
        }
      }
    }
    "ratio" => match cw_tokens_raw {
      Some(context_window_tokens) => {
        let approx_total_tokens = approx_token_count_from_byte_len(total_with_extra);
        let approx_ratio = if context_window_tokens == 0 {
          1.0
        } else {
          (approx_total_tokens as f32) / (context_window_tokens as f32)
        };
        if approx_ratio < hs.trigger_on_context_ratio {
          TriggerDecision::NotTriggered
        } else {
          let threshold_tokens =
            ((context_window_tokens as f64) * (hs.trigger_on_context_ratio as f64)).ceil() as u64;
          let threshold_chars = u64::try_from(threshold_tokens)
            .unwrap_or(u64::MAX)
            .saturating_mul(4) as usize;

          let target_tokens =
            ((context_window_tokens as f64) * (hs.target_context_ratio as f64)).floor() as u64;
          let target_chars_budget = u64::try_from(target_tokens)
            .unwrap_or(u64::MAX)
            .saturating_mul(4) as usize;
          let summary_overhead = hs
            .abridged_history_params
            .total_chars_limit
            .saturating_add((hs.max_tokens as usize).saturating_mul(4))
            .saturating_add(4096);
          let target_tail_budget_chars = target_chars_budget.saturating_sub(summary_overhead);

          TriggerDecision::TriggerRatio {
            context_window_tokens,
            threshold_chars,
            target_tail_budget_chars,
            approx_total_tokens,
            approx_ratio,
          }
        }
      }
      None => {
        if total_with_extra < hs.trigger_on_history_size_chars {
          TriggerDecision::NotTriggered
        } else {
          TriggerDecision::TriggerChars {
            threshold_chars: hs.trigger_on_history_size_chars,
          }
        }
      }
    },
    "auto" | _ => {
      if let Some(context_window_tokens_raw) = cw_tokens_raw {
        let cap_tokens =
          u32::try_from(hs.trigger_on_history_size_chars.saturating_div(4)).unwrap_or(u32::MAX);
        let context_window_tokens = if cap_tokens > 0 {
          context_window_tokens_raw.min(cap_tokens)
        } else {
          context_window_tokens_raw
        };

        let approx_total_tokens = approx_token_count_from_byte_len(total_with_extra);
        let approx_ratio = if context_window_tokens == 0 {
          1.0
        } else {
          (approx_total_tokens as f32) / (context_window_tokens as f32)
        };
        if approx_ratio < hs.trigger_on_context_ratio {
          TriggerDecision::NotTriggered
        } else {
          let threshold_tokens =
            ((context_window_tokens as f64) * (hs.trigger_on_context_ratio as f64)).ceil() as u64;
          let threshold_chars = u64::try_from(threshold_tokens)
            .unwrap_or(u64::MAX)
            .saturating_mul(4) as usize;

          let target_tokens =
            ((context_window_tokens as f64) * (hs.target_context_ratio as f64)).floor() as u64;
          let target_chars_budget = u64::try_from(target_tokens)
            .unwrap_or(u64::MAX)
            .saturating_mul(4) as usize;
          let summary_overhead = hs
            .abridged_history_params
            .total_chars_limit
            .saturating_add((hs.max_tokens as usize).saturating_mul(4))
            .saturating_add(4096);
          let target_tail_budget_chars = target_chars_budget.saturating_sub(summary_overhead);

          TriggerDecision::TriggerRatio {
            context_window_tokens,
            threshold_chars,
            target_tail_budget_chars,
            approx_total_tokens,
            approx_ratio,
          }
        }
      } else if total_with_extra < hs.trigger_on_history_size_chars {
        TriggerDecision::NotTriggered
      } else {
        TriggerDecision::TriggerChars {
          threshold_chars: hs.trigger_on_history_size_chars,
        }
      }
    }
  };

  let (trigger_threshold_chars, tail_size_chars_to_exclude) = match decision {
    TriggerDecision::NotTriggered => return Ok(false),
    TriggerDecision::TriggerChars { threshold_chars } => {
      (threshold_chars, hs.history_tail_size_chars_to_exclude)
    }
    TriggerDecision::TriggerRatio {
      context_window_tokens,
      threshold_chars,
      target_tail_budget_chars,
      approx_total_tokens,
      approx_ratio,
    } => {
      debug!(
        conversation_id=%conv_id,
        model=%chat_model,
        context_window_tokens=context_window_tokens,
        approx_total_tokens=approx_total_tokens,
        approx_ratio=approx_ratio,
        "history_summary 触发（ratio）"
      );
      (threshold_chars, target_tail_budget_chars)
    }
  };

  let split = split_history_for_summary(
    &augment.chat_history,
    tail_size_chars_to_exclude,
    trigger_threshold_chars,
    hs.min_tail_exchanges,
  );
  if split.head.is_empty() || split.tail.is_empty() {
    return Ok(false);
  }

  let split_boundary_request_id = split
    .tail
    .first()
    .map(|h| h.request_id.clone())
    .unwrap_or_default();
  if split_boundary_request_id.trim().is_empty() {
    return Ok(false);
  }

  let tail_start = augment
    .chat_history
    .iter()
    .position(|h| h.request_id == split_boundary_request_id)
    .unwrap_or(augment.chat_history.len().saturating_sub(1));
  let tail_start = adjust_tail_to_avoid_tool_result_orphans(&augment.chat_history, tail_start);

  let boundary_request_id = augment
    .chat_history
    .get(tail_start)
    .map(|h| h.request_id.clone())
    .unwrap_or_default();
  if boundary_request_id.trim().is_empty() {
    return Ok(false);
  }

  let dropped_head: Vec<AugmentChatHistory> = augment.chat_history[..tail_start].to_vec();
  let tail: Vec<AugmentChatHistory> = augment.chat_history[tail_start..].to_vec();
  if dropped_head.is_empty() || tail.is_empty() {
    return Ok(false);
  }

  let (abridged_history_text, num_dropped_in_beginning) = build_abridged_history_text(
    &augment.chat_history,
    &hs.abridged_history_params,
    Some(boundary_request_id.as_str()),
  );

  let now = now_ms();
  let (summary_text, summarization_request_id) = match cache.read().await.get_fresh(
    conv_id,
    boundary_request_id.as_str(),
    now,
    hs.cache_ttl_ms,
  ) {
    Some((s, id)) => {
      debug!(conversation_id=%conv_id, boundary_request_id=%boundary_request_id, "history_summary 命中缓存");
      (s, id)
    }
    None => {
      let provider_id = if hs.provider_id.trim().is_empty() {
        chat_provider_id
      } else {
        hs.provider_id.as_str()
      };
      let provider = get_byok_provider_by_id(cfg, provider_id)
        .context("history_summary.provider_id 无效")?;

      let mut used_rolling = false;
      let mut prompt = hs.prompt.clone();
      let mut input_history = dropped_head.clone();

      if hs.rolling_summary {
        if let Some(prev) = cache
          .read()
          .await
          .get_fresh_state(conv_id, now, hs.cache_ttl_ms)
        {
          if prev.summarized_until_request_id != boundary_request_id {
            let prev_boundary_pos = augment
              .chat_history
              .iter()
              .position(|h| h.request_id == prev.summarized_until_request_id);
            if let Some(pos) = prev_boundary_pos.filter(|p| *p < tail_start) {
              let mut delta = augment.chat_history[pos..tail_start].to_vec();
              if !delta.is_empty() {
                let prev_exchange = AugmentChatHistory {
                  response_text: String::new(),
                  request_message: format!(
                    "[PREVIOUS_SUMMARY]\n{}\n[/PREVIOUS_SUMMARY]",
                    prev.summary_text.trim()
                  ),
                  request_id: "proxy_history_summary_prev".to_string(),
                  request_nodes: Vec::new(),
                  structured_request_nodes: Vec::new(),
                  nodes: Vec::new(),
                  response_nodes: Vec::new(),
                  structured_output_nodes: Vec::new(),
                };
                let mut merged = Vec::with_capacity(1 + delta.len());
                merged.push(prev_exchange);
                merged.append(&mut delta);
                input_history = merged;
                used_rolling = true;
                prompt = format!(
                  "{}\n\nYou will be given an existing summary and additional new conversation turns. Update the summary to include the new information. Output only the updated summary.",
                  hs.prompt.trim()
                );
                debug!(
                  conversation_id=%conv_id,
                  prev_boundary_request_id=%prev.summarized_until_request_id,
                  new_boundary_request_id=%boundary_request_id,
                  "history_summary 使用滚动摘要（增量更新）"
                );
              }
            }
          }
        }
      }

      if hs.max_summarization_input_chars > 0 {
        if used_rolling {
          while input_history.len() > 1
            && estimate_history_size_chars(&input_history) > hs.max_summarization_input_chars
          {
            input_history.remove(1);
          }
        } else {
          while !input_history.is_empty()
            && estimate_history_size_chars(&input_history) > hs.max_summarization_input_chars
          {
            input_history.remove(0);
          }
        }
      }
      if input_history.is_empty() {
        return Ok(false);
      }

      let default_model = match provider {
        SummaryProviderRef::Anthropic(p) => p.default_model.as_str(),
        SummaryProviderRef::OpenAICompatible(p) => p.default_model.as_str(),
      };
      let model = if !hs.model.trim().is_empty() {
        hs.model.trim().to_string()
      } else if hs.provider_id.trim().is_empty() && !chat_model.trim().is_empty() {
        chat_model.trim().to_string()
      } else {
        default_model.trim().to_string()
      };

      let (req_id, text) = run_summary_model_once(
        http,
        provider,
        prompt.as_str(),
        input_history,
        hs.max_tokens,
        hs.timeout_seconds,
        model,
      )
      .await
      .context("history_summary 摘要模型调用失败")?;

      let req_id = if req_id.trim().is_empty() {
        format!("proxy_history_summary_{}", now)
      } else {
        req_id
      };
      let text = text.trim().to_string();
      if text.is_empty() {
        return Ok(false);
      }

      let snapshot = {
        let mut guard = cache.write().await;
        guard.put(
          conv_id,
          boundary_request_id.as_str(),
          text.clone(),
          req_id.clone(),
          now,
        );
        guard.clone()
      };
      if let Err(err) = snapshot.save_to_file(cache_path).await {
        debug!(error=%err, cache_path=%cache_path.display(), "history_summary cache 持久化失败（已忽略）");
      }
      (text, req_id)
    }
  };

  let template = hs.summary_node_request_message_template.clone();
  let history_end = build_history_end_exchanges(&tail);

  let summary_node = serde_json::json!({
    "summary_text": summary_text,
    "summarization_request_id": summarization_request_id,
    "history_beginning_dropped_num_exchanges": num_dropped_in_beginning,
    "history_middle_abridged_text": abridged_history_text,
    "history_end": history_end,
    "message_template": template,
  });

  let mut extra_tool_results: Vec<NodeIn> = Vec::new();
  extra_tool_results.extend(take_tool_result_nodes(&mut augment.request_nodes));
  extra_tool_results.extend(take_tool_result_nodes(
    &mut augment.structured_request_nodes,
  ));
  extra_tool_results.extend(take_tool_result_nodes(&mut augment.nodes));

  let mut summary_req_nodes: Vec<NodeIn> = Vec::with_capacity(1 + extra_tool_results.len());
  summary_req_nodes.push(build_history_summary_node(-10, summary_node));
  summary_req_nodes.extend(extra_tool_results);

  let summary_item = AugmentChatHistory {
    response_text: String::new(),
    request_message: String::new(),
    request_id: "proxy_history_summary".to_string(),
    request_nodes: summary_req_nodes,
    structured_request_nodes: Vec::new(),
    nodes: Vec::new(),
    response_nodes: Vec::new(),
    structured_output_nodes: Vec::new(),
  };

  augment.chat_history = vec![summary_item];
  compact_chat_history(&mut augment.chat_history);
  let after_chars = estimate_history_size_chars(&augment.chat_history);
  info!(
    conversation_id=%conv_id,
    before_chars=total_chars,
    after_chars=after_chars,
    tail_start=tail_start,
    "history_summary 已在 proxy 侧应用（client 无感）"
  );

  Ok(true)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::protocol::{
    ToolResultContentNode, ToolResultNode, ToolUse, TOOL_RESULT_CONTENT_NODE_TEXT,
  };

  fn empty_node(id: i32, node_type: i32) -> NodeIn {
    NodeIn {
      id,
      node_type,
      content: String::new(),
      text_node: None,
      tool_result_node: None,
      image_node: None,
      image_id_node: None,
      ide_state_node: None,
      edit_events_node: None,
      checkpoint_ref_node: None,
      change_personality_node: None,
      file_node: None,
      file_id_node: None,
      history_summary_node: None,
      tool_use: None,
      thinking: None,
    }
  }

  fn tool_use_node(tool_name: &str, input_json: &str) -> NodeIn {
    let mut n = empty_node(1, RESPONSE_NODE_TOOL_USE);
    n.tool_use = Some(ToolUse {
      tool_use_id: "tool-1".to_string(),
      tool_name: tool_name.to_string(),
      input_json: input_json.to_string(),
      mcp_server_name: String::new(),
      mcp_tool_name: String::new(),
    });
    n
  }

  fn tool_result_node() -> NodeIn {
    let mut n = empty_node(2, REQUEST_NODE_TOOL_RESULT);
    n.tool_result_node = Some(ToolResultNode {
      tool_use_id: "tool-1".to_string(),
      content: "OK".to_string(),
      content_nodes: vec![ToolResultContentNode {
        node_type: TOOL_RESULT_CONTENT_NODE_TEXT,
        text_content: "OK".to_string(),
        image_content: None,
      }],
      is_error: false,
    });
    n
  }

  #[test]
  fn infer_context_window_tokens_from_model_name_works() {
    assert_eq!(
      infer_context_window_tokens_from_model_name("gpt-4o-mini"),
      Some(128_000)
    );
    assert_eq!(
      infer_context_window_tokens_from_model_name("claude-3-5-sonnet-20241022"),
      Some(200_000)
    );
    assert_eq!(
      infer_context_window_tokens_from_model_name("gemini-2.5-pro"),
      Some(1_000_000)
    );
    assert_eq!(
      infer_context_window_tokens_from_model_name("llama-3.1-8b-instruct-128k"),
      Some(128_000)
    );
    assert_eq!(
      infer_context_window_tokens_from_model_name("qwen2.5-72b-instruct-32k"),
      Some(32 * 1024)
    );
  }

  #[test]
  fn resolve_context_window_tokens_falls_back_to_infer() {
    let hs = crate::config::HistorySummaryConfig::default();
    assert_eq!(resolve_context_window_tokens(&hs, "gpt-4o"), Some(128_000));
  }

  #[test]
  fn abridged_history_includes_actions_and_response() {
    let ex1 = AugmentChatHistory {
      response_text: "".to_string(),
      request_message: "view file".to_string(),
      request_id: "r1".to_string(),
      request_nodes: Vec::new(),
      structured_request_nodes: Vec::new(),
      nodes: Vec::new(),
      response_nodes: vec![tool_use_node("view", r#"{"path":"src/main.rs"}"#)],
      structured_output_nodes: Vec::new(),
    };
    let ex2 = AugmentChatHistory {
      response_text: "here is file".to_string(),
      request_message: "".to_string(),
      request_id: "r2".to_string(),
      request_nodes: vec![tool_result_node()],
      structured_request_nodes: Vec::new(),
      nodes: Vec::new(),
      response_nodes: Vec::new(),
      structured_output_nodes: Vec::new(),
    };

    let params = AbridgedHistoryParams::default();
    let (txt, _dropped) = build_abridged_history_text(&[ex1, ex2], &params, None);
    assert!(txt.contains("<agent_actions_summary>"));
    assert!(txt.contains("<files_viewed>"));
    assert!(txt.contains("src/main.rs"));
    assert!(txt.contains("<agent_response>"));
    assert!(txt.contains("here is file"));
  }

  #[test]
  fn split_history_respects_min_tail_exchanges() {
    let mk = |rid: &str| AugmentChatHistory {
      response_text: "a".to_string(),
      request_message: "b".to_string(),
      request_id: rid.to_string(),
      request_nodes: Vec::new(),
      structured_request_nodes: Vec::new(),
      nodes: Vec::new(),
      response_nodes: Vec::new(),
      structured_output_nodes: Vec::new(),
    };
    let history = vec![mk("r1"), mk("r2"), mk("r3"), mk("r4"), mk("r5")];
    let split = split_history_for_summary(&history, 0, 1, 2);
    assert!(split.tail.len() >= 2);
    assert_eq!(split.head.len() + split.tail.len(), history.len());
  }

  #[test]
  fn tail_adjustment_avoids_tool_result_as_first_tail_exchange() {
    let mk = |rid: &str, has_tool_result: bool| AugmentChatHistory {
      response_text: String::new(),
      request_message: rid.to_string(),
      request_id: rid.to_string(),
      request_nodes: if has_tool_result {
        vec![tool_result_node()]
      } else {
        Vec::new()
      },
      structured_request_nodes: Vec::new(),
      nodes: Vec::new(),
      response_nodes: Vec::new(),
      structured_output_nodes: Vec::new(),
    };
    let history = vec![mk("r1", false), mk("r2", true), mk("r3", false)];
    let start = adjust_tail_to_avoid_tool_result_orphans(&history, 1);
    assert_eq!(start, 0);
  }
}
