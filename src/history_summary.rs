use serde::Deserialize;
use serde_json::Value;

use crate::protocol::{
  AugmentChatHistory, NodeIn, TextNode, REQUEST_NODE_HISTORY_SUMMARY, REQUEST_NODE_TEXT,
  REQUEST_NODE_TOOL_RESULT, RESPONSE_NODE_RAW_RESPONSE, RESPONSE_NODE_THINKING, RESPONSE_NODE_TOOL_USE,
};

#[derive(Debug, Clone, Deserialize)]
struct HistorySummaryNode {
  #[serde(default, alias = "summaryText")]
  summary_text: String,
  #[serde(default, alias = "summarizationRequestId")]
  summarization_request_id: String,
  #[serde(default, alias = "historyBeginningDroppedNumExchanges")]
  history_beginning_dropped_num_exchanges: i64,
  #[serde(default, alias = "historyMiddleAbridgedText")]
  history_middle_abridged_text: String,
  #[serde(default, alias = "historyEnd")]
  history_end: Vec<HistoryEndExchange>,
  #[serde(default, alias = "messageTemplate")]
  message_template: String,
}

#[derive(Debug, Clone, Deserialize)]
struct HistoryEndExchange {
  #[serde(default, alias = "requestId")]
  request_id: String,
  #[serde(default, alias = "requestMessage")]
  request_message: String,
  #[serde(default, alias = "responseText")]
  response_text: String,
  #[serde(default, alias = "requestNodes")]
  request_nodes: Vec<NodeIn>,
  #[serde(default, alias = "responseNodes")]
  response_nodes: Vec<NodeIn>,
}

#[derive(Debug, Clone)]
struct ExchangeRenderCtx {
  user_message: String,
  tool_results: Vec<ToolResultCtx>,
  thinking: String,
  response_text: String,
  tool_uses: Vec<ToolUseCtx>,
  has_response: bool,
}

#[derive(Debug, Clone)]
struct ToolResultCtx {
  id: String,
  content: String,
  is_error: bool,
}

#[derive(Debug, Clone)]
struct ToolUseCtx {
  name: String,
  id: String,
  input: String,
}

fn normalize_joined_lines(lines: impl IntoIterator<Item = String>) -> String {
  let mut out = String::new();
  for line in lines {
    let line = line.trim_end_matches('\n');
    if line.is_empty() {
      continue;
    }
    if !out.is_empty() {
      out.push('\n');
    }
    out.push_str(line);
  }
  out
}

fn extract_user_message_from_request_nodes(nodes: &[NodeIn], fallback_request_message: &str) -> String {
  let joined = normalize_joined_lines(nodes.iter().filter_map(|n| {
    if n.node_type == REQUEST_NODE_TEXT {
      n.text_node.as_ref().map(|t| t.content.clone())
    } else {
      None
    }
  }));
  if !joined.trim().is_empty() {
    joined
  } else {
    fallback_request_message.to_string()
  }
}

fn build_exchange_render_ctx(ex: &HistoryEndExchange) -> ExchangeRenderCtx {
  let user_message = extract_user_message_from_request_nodes(&ex.request_nodes, &ex.request_message);

  let tool_results: Vec<ToolResultCtx> = ex
    .request_nodes
    .iter()
    .filter(|n| n.node_type == REQUEST_NODE_TOOL_RESULT)
    .filter_map(|n| {
      let tr = n.tool_result_node.as_ref()?;
      if tr.tool_use_id.trim().is_empty() {
        return None;
      }
      Some(ToolResultCtx {
        id: tr.tool_use_id.clone(),
        content: tr.content.clone(),
        is_error: tr.is_error,
      })
    })
    .collect();

  let thinking = normalize_joined_lines(ex.response_nodes.iter().filter_map(|n| {
    if n.node_type == RESPONSE_NODE_THINKING {
      n.thinking
        .as_ref()
        .map(|t| t.summary.clone())
        .filter(|s| !s.trim().is_empty())
    } else {
      None
    }
  }));

  let response_text = normalize_joined_lines(ex.response_nodes.iter().filter_map(|n| {
    if n.node_type == RESPONSE_NODE_RAW_RESPONSE && !n.content.trim().is_empty() {
      Some(n.content.clone())
    } else {
      None
    }
  }));

  let tool_uses: Vec<ToolUseCtx> = ex
    .response_nodes
    .iter()
    .filter(|n| n.node_type == RESPONSE_NODE_TOOL_USE)
    .filter_map(|n| {
      let tu = n.tool_use.as_ref()?;
      if tu.tool_use_id.trim().is_empty() || tu.tool_name.trim().is_empty() {
        return None;
      }
      Some(ToolUseCtx {
        name: tu.tool_name.clone(),
        id: tu.tool_use_id.clone(),
        input: tu.input_json.clone(),
      })
    })
    .collect();

  let has_response = !thinking.is_empty() || !response_text.is_empty() || !tool_uses.is_empty();

  ExchangeRenderCtx {
    user_message,
    tool_results,
    thinking,
    response_text,
    tool_uses,
    has_response,
  }
}

fn render_exchange_full(ctx: &ExchangeRenderCtx) -> String {
  let mut out = String::new();
  out.push_str("<exchange>\n  <user_request_or_tool_results>\n");
  if !ctx.user_message.trim().is_empty() {
    out.push_str(ctx.user_message.trim_end_matches('\n'));
    out.push('\n');
  }
  for tr in &ctx.tool_results {
    out.push_str(&format!(
      "    <tool_result tool_use_id=\"{}\" is_error=\"{}\">\n",
      tr.id.trim(),
      if tr.is_error { "true" } else { "false" }
    ));
    if !tr.content.trim().is_empty() {
      out.push_str(tr.content.trim_end_matches('\n'));
      out.push('\n');
    }
    out.push_str("    </tool_result>\n");
  }
  out.push_str("  </user_request_or_tool_results>\n");

  if ctx.has_response {
    out.push_str("  <agent_response_or_tool_uses>\n");
    if !ctx.thinking.trim().is_empty() {
      out.push_str("    <thinking>\n");
      out.push_str(ctx.thinking.trim_end_matches('\n'));
      out.push('\n');
      out.push_str("    </thinking>\n");
    }
    if !ctx.response_text.trim().is_empty() {
      out.push_str(ctx.response_text.trim_end_matches('\n'));
      out.push('\n');
    }
    for tu in &ctx.tool_uses {
      out.push_str(&format!(
        "    <tool_use name=\"{}\" tool_use_id=\"{}\">\n",
        tu.name.trim(),
        tu.id.trim()
      ));
      if !tu.input.trim().is_empty() {
        out.push_str(tu.input.trim_end_matches('\n'));
        out.push('\n');
      }
      out.push_str("    </tool_use>\n");
    }
    out.push_str("  </agent_response_or_tool_uses>\n");
  }

  out.push_str("</exchange>");
  out
}

fn replace_placeholders(mut template: String, repl: &[(&str, String)]) -> String {
  for (k, v) in repl {
    if template.contains(k) {
      template = template.replace(k, v);
    }
  }
  template
}

pub fn render_history_summary_node_value(v: &Value, extra_tool_results: &[NodeIn]) -> Option<String> {
  let mut node: HistorySummaryNode = serde_json::from_value(v.clone()).ok()?;
  if node.message_template.trim().is_empty() {
    return None;
  }

  if !extra_tool_results.is_empty() {
    node.history_end.push(HistoryEndExchange {
      request_id: String::new(),
      request_message: String::new(),
      response_text: String::new(),
      request_nodes: extra_tool_results.to_vec(),
      response_nodes: Vec::new(),
    });
  }

  let end_part_full = node
    .history_end
    .iter()
    .map(build_exchange_render_ctx)
    .map(|ctx| render_exchange_full(&ctx))
    .collect::<Vec<_>>()
    .join("\n");

  let abridged = node.history_middle_abridged_text.clone();
  let rendered = replace_placeholders(
    node.message_template.clone(),
    &[
      ("{summary}", node.summary_text),
      ("{summarization_request_id}", node.summarization_request_id),
      (
        "{beginning_part_dropped_num_exchanges}",
        node.history_beginning_dropped_num_exchanges.to_string(),
      ),
      ("{middle_part_abridged}", abridged.clone()),
      ("{end_part_full}", end_part_full),
      // 兼容旧模板字段名
      ("{abridged_history}", abridged),
    ],
  );

  Some(rendered)
}

fn has_history_summary_node(nodes: &[NodeIn]) -> bool {
  nodes.iter().any(|n| {
    n.node_type == REQUEST_NODE_HISTORY_SUMMARY && n.history_summary_node.is_some()
  })
}

fn chat_history_item_has_summary(item: &AugmentChatHistory) -> bool {
  has_history_summary_node(&item.request_nodes)
    || has_history_summary_node(&item.structured_request_nodes)
    || has_history_summary_node(&item.nodes)
}

pub fn compact_chat_history(chat_history: &mut Vec<AugmentChatHistory>) {
  let Some(start) = chat_history.iter().rposition(chat_history_item_has_summary) else {
    return;
  };

  if start > 0 {
    chat_history.drain(0..start);
  }

  let Some(first) = chat_history.first_mut() else {
    return;
  };

  let mut req_nodes: Vec<NodeIn> = Vec::new();
  req_nodes.append(&mut first.request_nodes);
  req_nodes.append(&mut first.structured_request_nodes);
  req_nodes.append(&mut first.nodes);

  let Some(summary_pos) = req_nodes.iter().position(|n| {
    n.node_type == REQUEST_NODE_HISTORY_SUMMARY && n.history_summary_node.is_some()
  }) else {
    first.request_nodes = req_nodes;
    return;
  };

  let summary_id = req_nodes[summary_pos].id;
  let summary_value = req_nodes[summary_pos]
    .history_summary_node
    .clone()
    .unwrap_or(Value::Null);

  let tool_results: Vec<NodeIn> = req_nodes
    .iter()
    .filter(|n| n.node_type == REQUEST_NODE_TOOL_RESULT && n.tool_result_node.is_some())
    .cloned()
    .collect();

  let mut other_nodes: Vec<NodeIn> = req_nodes
    .into_iter()
    .filter(|n| n.node_type != REQUEST_NODE_HISTORY_SUMMARY && n.node_type != REQUEST_NODE_TOOL_RESULT)
    .collect();

  let Some(text) = render_history_summary_node_value(&summary_value, &tool_results) else {
    // 无法渲染时，不做破坏性改写；仅保留裁剪 chat_history 的行为。
    first.request_nodes = other_nodes;
    return;
  };

  let summary_text_node = NodeIn {
    id: summary_id,
    node_type: REQUEST_NODE_TEXT,
    content: String::new(),
    text_node: Some(TextNode { content: text }),
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
  };

  let mut new_nodes = Vec::with_capacity(1 + other_nodes.len());
  new_nodes.push(summary_text_node);
  new_nodes.append(&mut other_nodes);
  first.request_nodes = new_nodes;
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::protocol::{ToolResultNode, TOOL_RESULT_CONTENT_NODE_TEXT};

  #[test]
  fn renders_history_summary_template_new() {
    let v = serde_json::json!({
      "summary_text": "SUM",
      "summarization_request_id": "req_123",
      "history_beginning_dropped_num_exchanges": 7,
      "history_middle_abridged_text": "<exchange>abridged</exchange>",
      "history_end": [
        {
          "request_id": "r1",
          "request_message": "hello",
          "response_text": "world",
          "request_nodes": [],
          "response_nodes": [
            { "id": 1, "type": 0, "content": "world" }
          ]
        }
      ],
      "message_template": r#"<supervisor>
<summary request_id="{summarization_request_id}">
{summary}
</summary>
Beginning part has {beginning_part_dropped_num_exchanges} exchanges.
<middle_part_abridged>
{middle_part_abridged}
</middle_part_abridged>
<end_part_full>
{end_part_full}
</end_part_full>
</supervisor>"#
    });

    let rendered = render_history_summary_node_value(&v, &[]).expect("should render");
    assert!(rendered.contains(r#"<summary request_id="req_123">"#));
    assert!(rendered.contains("SUM"));
    assert!(rendered.contains("Beginning part has 7 exchanges."));
    assert!(rendered.contains("<middle_part_abridged>"));
    assert!(rendered.contains("<exchange>abridged</exchange>"));
    assert!(rendered.contains("<end_part_full>"));
    assert!(rendered.contains("<exchange>"));
    assert!(rendered.contains("hello"));
    assert!(rendered.contains("world"));
  }

  #[test]
  fn compacts_chat_history_and_embeds_tool_results() {
    let summary = serde_json::json!({
      "summary_text": "S",
      "summarization_request_id": "req",
      "history_beginning_dropped_num_exchanges": 1,
      "history_middle_abridged_text": "",
      "history_end": [],
      "message_template": r#"<supervisor><end_part_full>{end_part_full}</end_part_full></supervisor>"#
    });

    let tool_result = NodeIn {
      id: 2,
      node_type: REQUEST_NODE_TOOL_RESULT,
      content: String::new(),
      text_node: None,
      tool_result_node: Some(ToolResultNode {
        tool_use_id: "tool-1".to_string(),
        content: "RESULT".to_string(),
        content_nodes: vec![crate::protocol::ToolResultContentNode {
          node_type: TOOL_RESULT_CONTENT_NODE_TEXT,
          text_content: "RESULT".to_string(),
          image_content: None,
        }],
        is_error: false,
      }),
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
    };

    let summary_node = NodeIn {
      id: 1,
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
      history_summary_node: Some(summary),
      tool_use: None,
      thinking: None,
    };

    let mut chat_history = vec![
      AugmentChatHistory {
        response_text: "old".to_string(),
        request_message: "old".to_string(),
        request_id: "r0".to_string(),
        request_nodes: Vec::new(),
        structured_request_nodes: Vec::new(),
        nodes: Vec::new(),
        response_nodes: Vec::new(),
        structured_output_nodes: Vec::new(),
      },
      AugmentChatHistory {
        response_text: "".to_string(),
        request_message: "".to_string(),
        request_id: "r1".to_string(),
        request_nodes: vec![summary_node, tool_result],
        structured_request_nodes: Vec::new(),
        nodes: Vec::new(),
        response_nodes: Vec::new(),
        structured_output_nodes: Vec::new(),
      },
    ];

    compact_chat_history(&mut chat_history);

    assert_eq!(chat_history.len(), 1);
    assert_eq!(chat_history[0].request_nodes.len() >= 1, true);
    assert_eq!(chat_history[0].request_nodes[0].node_type, REQUEST_NODE_TEXT);
    let txt = chat_history[0]
      .request_nodes[0]
      .text_node
      .as_ref()
      .map(|t| t.content.clone())
      .unwrap_or_default();
    assert!(txt.contains("tool_result"));
    assert!(txt.contains("tool-1"));
    assert!(txt.contains("RESULT"));
    assert!(
      !chat_history[0]
        .request_nodes
        .iter()
        .any(|n| n.node_type == REQUEST_NODE_TOOL_RESULT),
      "tool_result nodes should be removed from request_nodes"
    );
    assert!(
      !chat_history[0]
        .request_nodes
        .iter()
        .any(|n| n.node_type == REQUEST_NODE_HISTORY_SUMMARY),
      "history_summary node should be removed from request_nodes"
    );
  }
}
