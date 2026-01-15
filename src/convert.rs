use std::collections::{HashMap, HashSet};

use anyhow::Context;
use serde_json::Value;

use crate::{
  anthropic::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicRequest,
    AnthropicThinking, AnthropicTool, AnthropicToolChoice,
  },
  config::{AnthropicProviderConfig, OpenAICompatibleProviderConfig},
  openai::{
    OpenAIChatCompletionRequest, OpenAIChatMessage, OpenAIFunctionCall, OpenAIStreamOptions,
    OpenAITool, OpenAIToolCall,
  },
	  protocol::{
	    AugmentChatHistory, AugmentRequest, NodeIn, NodeOut, ThinkingNode, TokenUsageNode,
	    ToolDefinition, ToolResultContentNode, ToolUse, REQUEST_NODE_CHANGE_PERSONALITY,
	    REQUEST_NODE_CHECKPOINT_REF, REQUEST_NODE_EDIT_EVENTS, REQUEST_NODE_FILE, REQUEST_NODE_FILE_ID,
	    REQUEST_NODE_HISTORY_SUMMARY, REQUEST_NODE_IDE_STATE, REQUEST_NODE_IMAGE,
	    REQUEST_NODE_IMAGE_ID, REQUEST_NODE_TEXT, REQUEST_NODE_TOOL_RESULT,
	    RESPONSE_NODE_MAIN_TEXT_FINISHED, RESPONSE_NODE_RAW_RESPONSE, RESPONSE_NODE_THINKING,
	    RESPONSE_NODE_TOKEN_USAGE, RESPONSE_NODE_TOOL_USE, RESPONSE_NODE_TOOL_USE_START,
	    STOP_REASON_END_TURN, STOP_REASON_MAX_TOKENS, STOP_REASON_RECITATION, STOP_REASON_SAFETY,
	    STOP_REASON_TOOL_USE_REQUESTED,
	    TOOL_RESULT_CONTENT_NODE_IMAGE, TOOL_RESULT_CONTENT_NODE_TEXT,
	  },
	};

pub fn clean_model(model: &str) -> String {
  let model = model.trim();
  if let Some(stripped) = model.strip_prefix("gemini-") {
    if stripped.starts_with("claude-") {
      return stripped.to_string();
    }
  }
  model.to_string()
}

fn is_user_placeholder_message(message: &str) -> bool {
  let message = message.trim();
  if message.is_empty() {
    return false;
  }
  let mut count = 0usize;
  for ch in message.chars() {
    if ch != '-' {
      return false;
    }
    count += 1;
    if count > 16 {
      return false;
    }
  }
  true
}

fn push_text_block(
  blocks: &mut Vec<AnthropicContentBlock>,
  last_text: &mut Option<String>,
  text: &str,
) {
  let trimmed = text.trim();
  if trimmed.is_empty() || is_user_placeholder_message(trimmed) {
    return;
  }
  if last_text.as_deref() == Some(trimmed) {
    return;
  }
  blocks.push(text_block(text));
  *last_text = Some(trimmed.to_string());
}

pub fn convert_augment_to_anthropic(
  provider: &AnthropicProviderConfig,
  augment: &AugmentRequest,
  model: String,
) -> anyhow::Result<AnthropicRequest> {
  let system = build_system_prompt(augment);
  let mut messages: Vec<AnthropicMessage> = Vec::new();

  for (index, history) in augment.chat_history.iter().enumerate() {
    push_history_messages(
      &mut messages,
      augment.chat_history.as_slice(),
      index,
      history,
    )?;
  }

  let current_nodes = augment
    .nodes
    .iter()
    .chain(&augment.structured_request_nodes)
    .chain(&augment.request_nodes);
  if !augment.message.is_empty() || current_nodes.clone().next().is_some() {
    let user_content = build_user_content_blocks(&augment.message, current_nodes, true)?;
    if !user_content.is_empty() {
      messages.push(AnthropicMessage {
        role: "user".to_string(),
        content: user_content,
      });
    }
  }

  let tools = convert_tools(&augment.tool_definitions)?;
  let tool_choice = (!tools.is_empty()).then(|| AnthropicToolChoice {
    choice_type: "auto".to_string(),
    name: None,
  });

  let thinking = provider.thinking.enabled.then(|| AnthropicThinking {
    thinking_type: "enabled".to_string(),
    budget_tokens: provider.thinking.budget_tokens,
  });

  Ok(AnthropicRequest {
    model,
    messages,
    max_tokens: provider.max_tokens,
    system: (!system.is_empty()).then_some(system),
    temperature: None,
    top_p: None,
    top_k: None,
    stop_sequences: None,
    stream: true,
    tools: (!tools.is_empty()).then_some(tools),
    tool_choice,
    thinking,
  })
}

#[derive(Debug, Clone)]
enum OpenAISegment {
  Text(String),
  Image { media_type: String, data: String },
}

fn push_openai_text_segment(
  segments: &mut Vec<OpenAISegment>,
  last_text: &mut Option<String>,
  text: &str,
) {
  let trimmed = text.trim();
  if trimmed.is_empty() || is_user_placeholder_message(trimmed) {
    return;
  }
  if last_text.as_deref() == Some(trimmed) {
    return;
  }
  segments.push(OpenAISegment::Text(text.to_string()));
  *last_text = Some(trimmed.to_string());
}

fn push_openai_text_segment_from_value(
  segments: &mut Vec<OpenAISegment>,
  last_text: &mut Option<String>,
  value: Option<&Value>,
  fmt: fn(&Value) -> String,
) {
  let Some(v) = value else { return };
  let text = fmt(v);
  if text.trim().is_empty() {
    return;
  }
  push_openai_text_segment(segments, last_text, &text);
}

fn build_openai_user_segments<'a>(
  message: &str,
  nodes: impl Iterator<Item = &'a NodeIn>,
) -> anyhow::Result<Vec<OpenAISegment>> {
  let mut segments: Vec<OpenAISegment> = Vec::new();
  let mut last_text: Option<String> = None;
  push_openai_text_segment(&mut segments, &mut last_text, message);

  for node in nodes {
    match node.node_type {
      REQUEST_NODE_TEXT => {
        if let Some(t) = &node.text_node {
          push_openai_text_segment(&mut segments, &mut last_text, &t.content);
        }
      }
      REQUEST_NODE_TOOL_RESULT => {}
      REQUEST_NODE_IMAGE => {
        if let Some(img) = &node.image_node {
          let data = img.image_data.trim();
          if !data.is_empty() {
            segments.push(OpenAISegment::Image {
              media_type: map_image_format_to_media_type(img.format).to_string(),
              data: data.to_string(),
            });
            last_text = None;
          }
        }
      }
      REQUEST_NODE_IMAGE_ID => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.image_id_node.as_ref(),
        format_image_id_for_prompt,
      ),
      REQUEST_NODE_IDE_STATE => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.ide_state_node.as_ref(),
        format_ide_state_for_prompt,
      ),
      REQUEST_NODE_EDIT_EVENTS => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.edit_events_node.as_ref(),
        format_edit_events_for_prompt,
      ),
      REQUEST_NODE_CHECKPOINT_REF => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.checkpoint_ref_node.as_ref(),
        format_checkpoint_ref_for_prompt,
      ),
      REQUEST_NODE_CHANGE_PERSONALITY => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.change_personality_node.as_ref(),
        format_change_personality_for_prompt,
      ),
      REQUEST_NODE_FILE => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.file_node.as_ref(),
        format_file_node_for_prompt,
      ),
      REQUEST_NODE_FILE_ID => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.file_id_node.as_ref(),
        format_file_id_for_prompt,
      ),
      REQUEST_NODE_HISTORY_SUMMARY => push_openai_text_segment_from_value(
        &mut segments,
        &mut last_text,
        node.history_summary_node.as_ref(),
        format_history_summary_for_prompt,
      ),
      _ => {}
    }
  }

  Ok(segments)
}

fn build_openai_message_content(segments: Vec<OpenAISegment>) -> Option<Value> {
  if segments.is_empty() {
    return None;
  }
  let has_image = segments
    .iter()
    .any(|s| matches!(s, OpenAISegment::Image { .. }));
  if !has_image {
    let mut parts: Vec<String> = Vec::new();
    for s in segments {
      let OpenAISegment::Text(t) = s else { continue };
      let t = t.trim();
      if !t.is_empty() {
        parts.push(t.to_string());
      }
    }
    let text = parts.join("\n\n").trim().to_string();
    if text.is_empty() {
      None
    } else {
      Some(Value::String(text))
    }
  } else {
    let mut out: Vec<Value> = Vec::new();
    let mut text_buf = String::new();
    let flush_text = |out: &mut Vec<Value>, text_buf: &mut String| {
      let t = text_buf.trim();
      if !t.is_empty() {
        out.push(serde_json::json!({ "type": "text", "text": t }));
      }
      text_buf.clear();
    };

    for s in segments {
      match s {
        OpenAISegment::Text(t) => {
          let t = t.trim();
          if t.is_empty() {
            continue;
          }
          if !text_buf.is_empty() {
            text_buf.push_str("\n\n");
          }
          text_buf.push_str(t);
        }
        OpenAISegment::Image { media_type, data } => {
          flush_text(&mut out, &mut text_buf);
          let data = data.trim();
          if data.is_empty() {
            continue;
          }
          out.push(serde_json::json!({
            "type": "image_url",
            "image_url": { "url": format!("data:{media_type};base64,{data}") }
          }));
        }
      }
    }
    flush_text(&mut out, &mut text_buf);
    if out.is_empty() {
      None
    } else {
      Some(Value::Array(out))
    }
  }
}

fn build_openai_tool_calls_from_output_nodes<'a>(
  nodes: impl Iterator<Item = &'a NodeIn>,
) -> Vec<OpenAIToolCall> {
  let mut tool_use_nodes: Vec<&NodeIn> = Vec::new();
  let mut tool_use_start_nodes: Vec<&NodeIn> = Vec::new();
  for node in nodes {
    if node.node_type == RESPONSE_NODE_TOOL_USE {
      tool_use_nodes.push(node);
    } else if node.node_type == RESPONSE_NODE_TOOL_USE_START {
      tool_use_start_nodes.push(node);
    }
  }
  let chosen = if tool_use_nodes.is_empty() {
    tool_use_start_nodes
  } else {
    tool_use_nodes
  };
  let mut seen_ids: HashSet<String> = HashSet::new();
  let mut out: Vec<OpenAIToolCall> = Vec::new();
  for node in chosen {
    let Some(tool_use) = &node.tool_use else {
      continue;
    };
    if tool_use.tool_name.trim().is_empty() {
      continue;
    }
    let mut id = tool_use.tool_use_id.trim().to_string();
    if id.is_empty() {
      id = format!("tool-{}", out.len() + 1);
    }
    if !seen_ids.insert(id.clone()) {
      continue;
    }
    let args = if tool_use.input_json.trim().is_empty() {
      "{}"
    } else {
      tool_use.input_json.trim()
    };
    out.push(OpenAIToolCall {
      id,
      call_type: "function".to_string(),
      function: OpenAIFunctionCall {
        name: tool_use.tool_name.clone(),
        arguments: args.to_string(),
      },
    });
  }
  out
}

fn build_openai_tool_result_text(fallback_text: &str, nodes: &[ToolResultContentNode]) -> String {
  let mut parts: Vec<String> = Vec::new();
  let mut last_text = String::new();
  for n in nodes {
    match n.node_type {
      TOOL_RESULT_CONTENT_NODE_TEXT => {
        let text = n.text_content.trim();
        if text.is_empty() || is_user_placeholder_message(text) {
          continue;
        }
        if !last_text.is_empty() && last_text == text {
          continue;
        }
        parts.push(text.to_string());
        last_text = text.to_string();
      }
      TOOL_RESULT_CONTENT_NODE_IMAGE => {
        let Some(img) = &n.image_content else {
          continue;
        };
        let data = img.image_data.trim();
        if data.is_empty() {
          continue;
        }
        parts.push(format!(
          "[image omitted: format={} bytes≈{}]",
          img.format,
          (data.len() * 3) / 4
        ));
        last_text.clear();
      }
      _ => {}
    }
  }
  if !parts.is_empty() {
    parts.join("\n\n").trim().to_string()
  } else {
    fallback_text.trim().to_string()
  }
}

fn build_openai_tool_messages_from_request_nodes<'a>(
  nodes: impl Iterator<Item = &'a NodeIn>,
) -> Vec<OpenAIChatMessage> {
  let mut out: Vec<OpenAIChatMessage> = Vec::new();
  for node in nodes {
    if node.node_type != REQUEST_NODE_TOOL_RESULT {
      continue;
    }
    let Some(tool) = &node.tool_result_node else {
      continue;
    };
    let tool_use_id = tool.tool_use_id.trim();
    if tool_use_id.is_empty() {
      continue;
    }
    let content = build_openai_tool_result_text(tool.content.as_str(), &tool.content_nodes);
    out.push(OpenAIChatMessage {
      role: "tool".to_string(),
      content: Some(Value::String(content)),
      tool_calls: None,
      tool_call_id: Some(tool_use_id.to_string()),
    });
  }
  out
}

fn convert_openai_tools(defs: &[ToolDefinition]) -> anyhow::Result<Vec<OpenAITool>> {
  let mut tools: Vec<OpenAITool> = Vec::with_capacity(defs.len());
  for def in defs {
    let schema = if let Some(v) = &def.input_schema {
      v.clone()
    } else if !def.input_schema_json.trim().is_empty() {
      serde_json::from_str(&def.input_schema_json)
        .with_context(|| format!("解析 tool input_schema_json 失败: {}", def.name))?
    } else {
      serde_json::json!({"type":"object","properties":{}})
    };
    tools.push(OpenAITool {
      tool_type: "function".to_string(),
      function: crate::openai::OpenAIFunctionDefinition {
        name: def.name.clone(),
        description: (!def.description.trim().is_empty()).then_some(def.description.clone()),
        parameters: schema,
      },
    });
  }
  Ok(tools)
}

fn push_history_messages_openai(
  out: &mut Vec<OpenAIChatMessage>,
  all: &[AugmentChatHistory],
  index: usize,
  history: &AugmentChatHistory,
) -> anyhow::Result<()> {
  let req_nodes = history
    .request_nodes
    .iter()
    .chain(&history.structured_request_nodes)
    .chain(&history.nodes);
  let req_segments = build_openai_user_segments(&history.request_message, req_nodes)?;
  if let Some(content) = build_openai_message_content(req_segments) {
    out.push(OpenAIChatMessage {
      role: "user".to_string(),
      content: Some(content),
      tool_calls: None,
      tool_call_id: None,
    });
  }

  let out_nodes = history
    .response_nodes
    .iter()
    .chain(&history.structured_output_nodes);
  let assistant_text = if history.response_text.trim().is_empty() {
    extract_assistant_text_from_output_nodes(out_nodes.clone())
  } else {
    history.response_text.clone()
  };
  let tool_calls = build_openai_tool_calls_from_output_nodes(out_nodes.clone());
  let content =
    (!assistant_text.trim().is_empty()).then(|| Value::String(assistant_text.trim().to_string()));
  if content.is_some() || !tool_calls.is_empty() {
    out.push(OpenAIChatMessage {
      role: "assistant".to_string(),
      content,
      tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
      tool_call_id: None,
    });
  }

  if let Some(next) = all.get(index + 1) {
    let next_req_nodes = next
      .request_nodes
      .iter()
      .chain(&next.structured_request_nodes)
      .chain(&next.nodes);
    out.extend(build_openai_tool_messages_from_request_nodes(
      next_req_nodes,
    ));
  }
  Ok(())
}

pub fn convert_augment_to_openai_compatible(
  provider: &OpenAICompatibleProviderConfig,
  augment: &AugmentRequest,
  model: String,
) -> anyhow::Result<OpenAIChatCompletionRequest> {
  let mut messages: Vec<OpenAIChatMessage> = Vec::new();

  let system = build_system_prompt(augment);
  if !system.trim().is_empty() {
    messages.push(OpenAIChatMessage {
      role: "system".to_string(),
      content: Some(Value::String(system)),
      tool_calls: None,
      tool_call_id: None,
    });
  }

  for (index, history) in augment.chat_history.iter().enumerate() {
    push_history_messages_openai(
      &mut messages,
      augment.chat_history.as_slice(),
      index,
      history,
    )?;
  }

  let mut current_nodes: Vec<&NodeIn> = augment
    .nodes
    .iter()
    .chain(&augment.structured_request_nodes)
    .chain(&augment.request_nodes)
    .collect();

  messages.extend(build_openai_tool_messages_from_request_nodes(
    current_nodes.iter().copied(),
  ));
  current_nodes.retain(|n| n.node_type != REQUEST_NODE_TOOL_RESULT);

  let req_segments = build_openai_user_segments(&augment.message, current_nodes.iter().copied())?;
  if let Some(content) = build_openai_message_content(req_segments) {
    messages.push(OpenAIChatMessage {
      role: "user".to_string(),
      content: Some(content),
      tool_calls: None,
      tool_call_id: None,
    });
  }

  let tools = convert_openai_tools(&augment.tool_definitions)?;
  let tool_choice = (!tools.is_empty()).then(|| Value::String("auto".to_string()));

  Ok(OpenAIChatCompletionRequest {
    model,
    messages,
    stream: true,
    stream_options: Some(OpenAIStreamOptions {
      include_usage: true,
    }),
    max_tokens: Some(provider.max_tokens),
    temperature: None,
    top_p: None,
    tools: (!tools.is_empty()).then_some(tools),
    tool_choice,
  })
}

fn build_system_prompt(augment: &AugmentRequest) -> String {
  let mut parts: Vec<String> = Vec::new();
  if !augment.prefix.trim().is_empty() {
    parts.push(augment.prefix.trim().to_string());
  }
  if !augment.user_guidelines.trim().is_empty() {
    parts.push(augment.user_guidelines.trim().to_string());
  }
  if !augment.agent_memories.trim().is_empty() {
    parts.push(augment.agent_memories.trim().to_string());
  }
  if augment.mode.trim() == "AGENT" {
    parts.push(
			"You are an AI coding assistant with access to tools. Use tools when needed to complete tasks."
				.to_string(),
		);
  }
  if !augment.lang.trim().is_empty() {
    parts.push(format!(
      "The user is working with {} code.",
      augment.lang.trim()
    ));
  }
  if !augment.path.trim().is_empty() {
    parts.push(format!("Current file path: {}", augment.path.trim()));
  }
  parts.join("\n\n")
}

fn push_history_messages(
  out: &mut Vec<AnthropicMessage>,
  all: &[AugmentChatHistory],
  index: usize,
  history: &AugmentChatHistory,
) -> anyhow::Result<()> {
  let req_nodes = history
    .request_nodes
    .iter()
    .chain(&history.structured_request_nodes)
    .chain(&history.nodes);
  let user_content = build_user_content_blocks(&history.request_message, req_nodes, false)?;
  if !user_content.is_empty() {
    out.push(AnthropicMessage {
      role: "user".to_string(),
      content: user_content,
    });
  }

  let out_nodes = history
    .response_nodes
    .iter()
    .chain(&history.structured_output_nodes);
  let assistant_text = if history.response_text.trim().is_empty() {
    extract_assistant_text_from_output_nodes(out_nodes.clone())
  } else {
    history.response_text.clone()
  };
  let assistant_content = build_assistant_content_blocks(&assistant_text, out_nodes)?;
  if !assistant_content.is_empty() {
    out.push(AnthropicMessage {
      role: "assistant".to_string(),
      content: assistant_content,
    });
  }

  if let Some(next) = all.get(index + 1) {
    let next_req_nodes = next
      .request_nodes
      .iter()
      .chain(&next.structured_request_nodes)
      .chain(&next.nodes);
    let tool_results = build_tool_results(next_req_nodes)?;
    if !tool_results.is_empty() {
      out.push(AnthropicMessage {
        role: "user".to_string(),
        content: tool_results,
      });
    }
  }

  Ok(())
}

fn build_user_content_blocks<'a>(
  message: &str,
  nodes: impl Iterator<Item = &'a NodeIn>,
  include_tool_results: bool,
) -> anyhow::Result<Vec<AnthropicContentBlock>> {
  let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
  let mut last_text: Option<String> = None;
  push_text_block(&mut blocks, &mut last_text, message);

  for node in nodes {
    match node.node_type {
      REQUEST_NODE_TEXT => {
        push_text_block_from_request_text_node(&mut blocks, &mut last_text, node)
      }
      REQUEST_NODE_TOOL_RESULT => {
        if include_tool_results {
          if let Some(tool) = &node.tool_result_node {
            let tool_use_id = tool.tool_use_id.trim();
            if tool_use_id.is_empty() {
              continue;
            }
            let content =
              build_anthropic_tool_result_content(tool.content.as_str(), &tool.content_nodes);
            blocks.push(AnthropicContentBlock {
              block_type: "tool_result".to_string(),
              text: None,
              source: None,
              id: None,
              name: None,
              input: None,
              tool_use_id: Some(tool_use_id.to_string()),
              content: Some(content),
              is_error: Some(tool.is_error),
              thinking: None,
              signature: None,
            });
            last_text = None;
          }
        }
      }
      REQUEST_NODE_IMAGE => {
        if let Some(img) = &node.image_node {
          blocks.push(image_block_from_format_and_data(
            img.format,
            &img.image_data,
          ));
          last_text = None;
        }
      }
      REQUEST_NODE_IMAGE_ID => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.image_id_node.as_ref(),
        format_image_id_for_prompt,
      ),
      REQUEST_NODE_IDE_STATE => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.ide_state_node.as_ref(),
        format_ide_state_for_prompt,
      ),
      REQUEST_NODE_EDIT_EVENTS => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.edit_events_node.as_ref(),
        format_edit_events_for_prompt,
      ),
      REQUEST_NODE_CHECKPOINT_REF => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.checkpoint_ref_node.as_ref(),
        format_checkpoint_ref_for_prompt,
      ),
      REQUEST_NODE_CHANGE_PERSONALITY => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.change_personality_node.as_ref(),
        format_change_personality_for_prompt,
      ),
      REQUEST_NODE_FILE => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.file_node.as_ref(),
        format_file_node_for_prompt,
      ),
      REQUEST_NODE_FILE_ID => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.file_id_node.as_ref(),
        format_file_id_for_prompt,
      ),
      REQUEST_NODE_HISTORY_SUMMARY => push_text_block_from_value(
        &mut blocks,
        &mut last_text,
        node.history_summary_node.as_ref(),
        format_history_summary_for_prompt,
      ),
      _ => {}
    }
  }

  Ok(blocks)
}

fn extract_assistant_text_from_output_nodes<'a>(nodes: impl Iterator<Item = &'a NodeIn>) -> String {
  let mut finished: Option<String> = None;
  let mut raw = String::new();
  for n in nodes {
    if n.node_type == RESPONSE_NODE_MAIN_TEXT_FINISHED && !n.content.trim().is_empty() {
      finished = Some(n.content.clone());
    } else if n.node_type == RESPONSE_NODE_RAW_RESPONSE && !n.content.is_empty() {
      raw.push_str(&n.content);
    }
  }
  finished.unwrap_or(raw).trim().to_string()
}

fn build_assistant_content_blocks<'a>(
  text: &str,
  nodes: impl Iterator<Item = &'a NodeIn>,
) -> anyhow::Result<Vec<AnthropicContentBlock>> {
  let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
  if !text.is_empty() {
    blocks.push(text_block(text));
  }

  let mut tool_use_nodes: Vec<&NodeIn> = Vec::new();
  let mut tool_use_start_nodes: Vec<&NodeIn> = Vec::new();
  for node in nodes {
    if node.node_type == RESPONSE_NODE_TOOL_USE {
      tool_use_nodes.push(node);
    } else if node.node_type == RESPONSE_NODE_TOOL_USE_START {
      tool_use_start_nodes.push(node);
    }
  }

  let chosen = if tool_use_nodes.is_empty() {
    tool_use_start_nodes
  } else {
    tool_use_nodes
  };

  let mut seen_tool_use_ids: HashSet<String> = HashSet::new();
  for node in chosen {
    let Some(tool_use) = &node.tool_use else {
      continue;
    };
    if tool_use.tool_use_id.trim().is_empty() || tool_use.tool_name.trim().is_empty() {
      continue;
    }
    if !seen_tool_use_ids.insert(tool_use.tool_use_id.clone()) {
      continue;
    }

    let input: serde_json::Value = if tool_use.input_json.trim().is_empty() {
      serde_json::Value::Object(serde_json::Map::new())
    } else {
      serde_json::from_str(&tool_use.input_json)
        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
    };

    blocks.push(AnthropicContentBlock {
      block_type: "tool_use".to_string(),
      text: None,
      source: None,
      id: Some(tool_use.tool_use_id.clone()),
      name: Some(tool_use.tool_name.clone()),
      input: Some(input),
      tool_use_id: None,
      content: None,
      is_error: None,
      thinking: None,
      signature: None,
    });
  }

  Ok(blocks)
}

fn build_tool_results<'a>(
  nodes: impl Iterator<Item = &'a NodeIn>,
) -> anyhow::Result<Vec<AnthropicContentBlock>> {
  let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
  for node in nodes {
    if node.node_type != REQUEST_NODE_TOOL_RESULT {
      continue;
    }
    let Some(tool) = &node.tool_result_node else {
      continue;
    };
    if tool.tool_use_id.trim().is_empty() {
      continue;
    }
    blocks.push(AnthropicContentBlock {
      block_type: "tool_result".to_string(),
      text: None,
      source: None,
      id: None,
      name: None,
      input: None,
      tool_use_id: Some(tool.tool_use_id.clone()),
      content: Some(build_anthropic_tool_result_content(
        tool.content.as_str(),
        &tool.content_nodes,
      )),
      is_error: Some(tool.is_error),
      thinking: None,
      signature: None,
    });
  }
  Ok(blocks)
}

fn text_block(text: &str) -> AnthropicContentBlock {
  AnthropicContentBlock {
    block_type: "text".to_string(),
    text: Some(text.to_string()),
    source: None,
    id: None,
    name: None,
    input: None,
    tool_use_id: None,
    content: None,
    is_error: None,
    thinking: None,
    signature: None,
  }
}

fn convert_tools(defs: &[ToolDefinition]) -> anyhow::Result<Vec<AnthropicTool>> {
  let mut tools = Vec::with_capacity(defs.len());
  for def in defs {
    let input_schema = if let Some(v) = &def.input_schema {
      v.clone()
    } else if !def.input_schema_json.trim().is_empty() {
      serde_json::from_str(&def.input_schema_json)
        .with_context(|| format!("解析 tool input_schema_json 失败: {}", def.name))?
    } else {
      serde_json::json!({"type":"object","properties":{}})
    };

    tools.push(AnthropicTool {
      name: def.name.clone(),
      description: (!def.description.trim().is_empty()).then_some(def.description.clone()),
      input_schema,
    });
  }
  Ok(tools)
}

pub fn map_anthropic_stop_reason_to_augment(reason: &str) -> i32 {
  match reason {
    "end_turn" => STOP_REASON_END_TURN,
    "max_tokens" => STOP_REASON_MAX_TOKENS,
    "tool_use" => STOP_REASON_TOOL_USE_REQUESTED,
    "stop_sequence" => STOP_REASON_END_TURN,
    "safety" => STOP_REASON_SAFETY,
    "recitation" => STOP_REASON_RECITATION,
    _ => STOP_REASON_END_TURN,
  }
}

pub fn map_openai_finish_reason_to_augment(reason: &str) -> i32 {
  match reason {
    "stop" => STOP_REASON_END_TURN,
    "length" => STOP_REASON_MAX_TOKENS,
    "tool_calls" => STOP_REASON_TOOL_USE_REQUESTED,
    "function_call" => STOP_REASON_TOOL_USE_REQUESTED,
    "content_filter" => STOP_REASON_SAFETY,
    _ => STOP_REASON_END_TURN,
  }
}

fn map_image_format_to_media_type(format: i32) -> &'static str {
  match format {
    2 => "image/jpeg",
    3 => "image/gif",
    4 => "image/webp",
    _ => "image/png",
  }
}

fn image_block_from_format_and_data(format: i32, data: &str) -> AnthropicContentBlock {
  AnthropicContentBlock {
    block_type: "image".to_string(),
    text: None,
    source: Some(AnthropicImageSource {
      source_type: "base64".to_string(),
      media_type: map_image_format_to_media_type(format).to_string(),
      data: data.to_string(),
    }),
    id: None,
    name: None,
    input: None,
    tool_use_id: None,
    content: None,
    is_error: None,
    thinking: None,
    signature: None,
  }
}

fn push_text_block_from_request_text_node(
  blocks: &mut Vec<AnthropicContentBlock>,
  last_text: &mut Option<String>,
  node: &NodeIn,
) {
  if let Some(t) = &node.text_node {
    push_text_block(blocks, last_text, &t.content);
  }
}

fn push_text_block_from_value(
  blocks: &mut Vec<AnthropicContentBlock>,
  last_text: &mut Option<String>,
  value: Option<&Value>,
  fmt: fn(&Value) -> String,
) {
  let Some(v) = value else { return };
  let text = fmt(v);
  if text.trim().is_empty() {
    return;
  }
  push_text_block(blocks, last_text, &text);
}

fn build_anthropic_tool_result_content(
  fallback_text: &str,
  nodes: &[ToolResultContentNode],
) -> Value {
  let mut content_blocks: Vec<Value> = Vec::new();
  let mut last_text = String::new();

  for n in nodes {
    match n.node_type {
      TOOL_RESULT_CONTENT_NODE_TEXT => {
        let text = n.text_content.trim();
        if text.is_empty() || is_user_placeholder_message(text) {
          continue;
        }
        if !last_text.is_empty() && last_text == text {
          continue;
        }
        content_blocks.push(serde_json::json!({ "type": "text", "text": text }));
        last_text = text.to_string();
      }
      TOOL_RESULT_CONTENT_NODE_IMAGE => {
        let Some(img) = &n.image_content else {
          continue;
        };
        let data = img.image_data.trim();
        if data.is_empty() {
          continue;
        }
        content_blocks.push(serde_json::json!({
          "type": "image",
          "source": { "type": "base64", "media_type": map_image_format_to_media_type(img.format), "data": data }
        }));
        last_text.clear();
      }
      _ => {}
    }
  }

  if !content_blocks.is_empty() {
    Value::Array(content_blocks)
  } else {
    Value::String(fallback_text.to_string())
  }
}

fn normalize_string(v: Option<&Value>) -> String {
  match v {
    Some(Value::String(s)) => s.trim().to_string(),
    _ => String::new(),
  }
}

fn truncate_inline_text(v: Option<&Value>, max_chars: usize) -> String {
  let s = normalize_string(v);
  if s.is_empty() {
    return s;
  }
  let mut out = String::new();
  for (i, ch) in s.chars().enumerate() {
    if i >= max_chars {
      out.push('…');
      break;
    }
    out.push(ch);
  }
  out
}

fn persona_type_to_label(v: Option<&Value>) -> &'static str {
  let n = v.and_then(|x| x.as_i64()).unwrap_or(0);
  match n {
    1 => "PROTOTYPER",
    2 => "BRAINSTORM",
    3 => "REVIEWER",
    _ => "DEFAULT",
  }
}

fn format_ide_state_for_prompt(v: &Value) -> String {
  let Some(ide) = v.as_object() else {
    return String::new();
  };
  let mut lines: Vec<String> = vec!["[IDE_STATE]".to_string()];

  let unchanged = ide
    .get("workspace_folders_unchanged")
    .or_else(|| ide.get("workspaceFoldersUnchanged"));
  if let Some(b) = unchanged.and_then(|x| x.as_bool()) {
    lines.push(format!("workspace_folders_unchanged={b}"));
  }

  let folders = ide
    .get("workspace_folders")
    .or_else(|| ide.get("workspaceFolders"))
    .and_then(|x| x.as_array())
    .cloned()
    .unwrap_or_default();
  if !folders.is_empty() {
    lines.push("workspace_folders:".to_string());
    for f in folders.iter().take(8) {
      let Some(r) = f.as_object() else { continue };
      let repo_root = truncate_inline_text(
        r.get("repository_root").or_else(|| r.get("repositoryRoot")),
        200,
      );
      let folder_root =
        truncate_inline_text(r.get("folder_root").or_else(|| r.get("folderRoot")), 200);
      if repo_root.is_empty() && folder_root.is_empty() {
        continue;
      }
      lines.push(format!(
        "- repository_root={} folder_root={}",
        if repo_root.is_empty() {
          "(unknown)"
        } else {
          repo_root.as_str()
        },
        if folder_root.is_empty() {
          "(unknown)"
        } else {
          folder_root.as_str()
        }
      ));
    }
  }

  if let Some(term) = ide
    .get("current_terminal")
    .or_else(|| ide.get("currentTerminal"))
    .and_then(|x| x.as_object())
  {
    let tid = term
      .get("terminal_id")
      .or_else(|| term.get("terminalId"))
      .and_then(|x| x.as_i64());
    let cwd = truncate_inline_text(
      term
        .get("current_working_directory")
        .or_else(|| term.get("currentWorkingDirectory")),
      200,
    );
    if tid.is_some() || !cwd.is_empty() {
      lines.push(format!(
        "current_terminal: id={} cwd={}",
        tid
          .map(|n| n.to_string())
          .unwrap_or_else(|| "?".to_string()),
        if cwd.is_empty() {
          "(unknown)"
        } else {
          cwd.as_str()
        }
      ));
    }
  }

  if lines.len() == 1 {
    return String::new();
  }
  lines.push("[/IDE_STATE]".to_string());
  lines.join("\n").trim().to_string()
}

fn format_edit_events_for_prompt(v: &Value) -> String {
  let Some(node) = v.as_object() else {
    return String::new();
  };
  let mut lines: Vec<String> = vec!["[EDIT_EVENTS]".to_string()];

  if let Some(src) = node.get("source") {
    if !src.is_null() {
      lines.push(format!("source={}", truncate_inline_text(Some(src), 200)));
    }
  }

  let events = node
    .get("edit_events")
    .or_else(|| node.get("editEvents"))
    .and_then(|x| x.as_array())
    .cloned()
    .unwrap_or_default();
  for ev in events.iter().take(6) {
    let Some(r) = ev.as_object() else { continue };
    let path = truncate_inline_text(r.get("path"), 200);
    let before_blob = truncate_inline_text(
      r.get("before_blob_name")
        .or_else(|| r.get("beforeBlobName")),
      120,
    );
    let after_blob = truncate_inline_text(
      r.get("after_blob_name").or_else(|| r.get("afterBlobName")),
      120,
    );
    let edits = r
      .get("edits")
      .and_then(|x| x.as_array())
      .cloned()
      .unwrap_or_default();

    lines.push(format!(
      "- file: {} edits={}{}{}",
      if path.is_empty() {
        "(unknown)"
      } else {
        path.as_str()
      },
      edits.len(),
      if before_blob.is_empty() {
        "".to_string()
      } else {
        format!(" before={before_blob}")
      },
      if after_blob.is_empty() {
        "".to_string()
      } else {
        format!(" after={after_blob}")
      }
    ));

    for ed in edits.iter().take(6) {
      let Some(e) = ed.as_object() else { continue };
      let after_start = e
        .get("after_line_start")
        .or_else(|| e.get("afterLineStart"))
        .and_then(|x| x.as_i64());
      let before_start = e
        .get("before_line_start")
        .or_else(|| e.get("beforeLineStart"))
        .and_then(|x| x.as_i64());
      let before_text =
        truncate_inline_text(e.get("before_text").or_else(|| e.get("beforeText")), 200);
      let after_text =
        truncate_inline_text(e.get("after_text").or_else(|| e.get("afterText")), 200);
      lines.push(format!(
        "  - edit: after_line_start={} before_line_start={} before=\"{}\" after=\"{}\"",
        after_start
          .map(|n| n.to_string())
          .unwrap_or_else(|| "?".to_string()),
        before_start
          .map(|n| n.to_string())
          .unwrap_or_else(|| "?".to_string()),
        before_text,
        after_text
      ));
    }
  }

  if lines.len() == 1 {
    return String::new();
  }
  lines.push("[/EDIT_EVENTS]".to_string());
  lines.join("\n").trim().to_string()
}

fn format_checkpoint_ref_for_prompt(v: &Value) -> String {
  let Some(r) = v.as_object() else {
    return String::new();
  };
  let mut lines: Vec<String> = vec!["[CHECKPOINT_REF]".to_string()];

  let request_id = truncate_inline_text(r.get("request_id").or_else(|| r.get("requestId")), 120);
  if !request_id.is_empty() {
    lines.push(format!("request_id={request_id}"));
  }

  let from = r
    .get("from_timestamp")
    .or_else(|| r.get("fromTimestamp"))
    .and_then(|x| x.as_i64());
  let to = r
    .get("to_timestamp")
    .or_else(|| r.get("toTimestamp"))
    .and_then(|x| x.as_i64());
  if from.is_some() || to.is_some() {
    lines.push(format!(
      "from_timestamp={} to_timestamp={}",
      from
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string()),
      to.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string())
    ));
  }

  if let Some(src) = r.get("source") {
    if !src.is_null() {
      lines.push(format!("source={}", truncate_inline_text(Some(src), 200)));
    }
  }

  if lines.len() == 1 {
    return String::new();
  }
  lines.push("[/CHECKPOINT_REF]".to_string());
  lines.join("\n").trim().to_string()
}

fn format_change_personality_for_prompt(v: &Value) -> String {
  let Some(p) = v.as_object() else {
    return String::new();
  };
  let t = persona_type_to_label(
    p.get("personality_type")
      .or_else(|| p.get("personalityType")),
  );
  let custom = truncate_inline_text(
    p.get("custom_instructions")
      .or_else(|| p.get("customInstructions")),
    1000,
  );
  let mut lines: Vec<String> = vec![
    "[CHANGE_PERSONALITY]".to_string(),
    format!("personality_type={t}"),
  ];
  if !custom.is_empty() {
    lines.push(format!("custom_instructions={custom}"));
  }
  lines.push("[/CHANGE_PERSONALITY]".to_string());
  lines.join("\n").trim().to_string()
}

fn format_image_id_for_prompt(v: &Value) -> String {
  let Some(img) = v.as_object() else {
    return String::new();
  };
  let id = truncate_inline_text(img.get("image_id").or_else(|| img.get("imageId")), 200);
  if id.is_empty() {
    return String::new();
  }
  let fmt = img.get("format").and_then(|x| x.as_i64());
  format!(
    "[IMAGE_ID] image_id={} format={}",
    id,
    fmt
      .map(|n| n.to_string())
      .unwrap_or_else(|| "?".to_string())
  )
}

fn format_file_id_for_prompt(v: &Value) -> String {
  let Some(f) = v.as_object() else {
    return String::new();
  };
  let id = truncate_inline_text(f.get("file_id").or_else(|| f.get("fileId")), 200);
  let name = truncate_inline_text(f.get("file_name").or_else(|| f.get("fileName")), 200);
  if id.is_empty() && name.is_empty() {
    return String::new();
  }
  format!(
    "[FILE_ID] file_name={} file_id={}",
    if name.is_empty() {
      "(unknown)"
    } else {
      name.as_str()
    },
    if id.is_empty() {
      "(unknown)"
    } else {
      id.as_str()
    }
  )
}

fn format_file_node_for_prompt(v: &Value) -> String {
  let Some(f) = v.as_object() else {
    return String::new();
  };
  let raw = normalize_string(f.get("file_data").or_else(|| f.get("fileData")));
  let format = normalize_string(f.get("format"));
  let format = if format.is_empty() {
    "application/octet-stream".to_string()
  } else {
    format
  };
  if raw.is_empty() {
    return format!("[FILE] format={format} (empty)");
  }

  let b64 = raw
    .strip_prefix("data:")
    .and_then(|rest| rest.split_once(";base64,").map(|(_, b)| b))
    .unwrap_or(raw.as_str());
  let approx_bytes = ((b64.len() * 3) / 4) as i64;

  let is_text_like = format.starts_with("text/")
    || matches!(
      format.as_str(),
      "application/json"
        | "application/xml"
        | "application/yaml"
        | "application/x-yaml"
        | "application/markdown"
    );
  if !is_text_like {
    return format!("[FILE] format={format} bytes≈{approx_bytes} (content omitted)");
  }

  let decoded = match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64) {
    Ok(v) => v,
    Err(_) => return format!("[FILE] format={format} bytes≈{approx_bytes} (decode failed)"),
  };
  let decoded = match String::from_utf8(decoded) {
    Ok(v) => v,
    Err(_) => return format!("[FILE] format={format} bytes≈{approx_bytes} (decode failed)"),
  };
  let max = 20_000usize;
  let content = if decoded.chars().count() > max {
    decoded.chars().take(max).collect::<String>() + "\n\n[Content truncated due to length...]"
  } else {
    decoded
  };
  format!("[FILE] format={format} bytes≈{approx_bytes}\n\n{content}")
    .trim()
    .to_string()
}

fn format_history_summary_for_prompt(v: &Value) -> String {
  if let Some(rendered) = crate::history_summary::render_history_summary_node_value(v, &[]) {
    return rendered;
  }
  let Some(h) = v.as_object() else {
    return String::new();
  };
  let summary_text =
    truncate_inline_text(h.get("summary_text").or_else(|| h.get("summaryText")), 3000);
  let req_id = truncate_inline_text(
    h.get("summarization_request_id")
      .or_else(|| h.get("summarizationRequestId")),
    120,
  );
  let dropped = h
    .get("history_beginning_dropped_num_exchanges")
    .or_else(|| h.get("historyBeginningDroppedNumExchanges"))
    .and_then(|x| x.as_i64());
  let abridged = truncate_inline_text(
    h.get("history_middle_abridged_text")
      .or_else(|| h.get("historyMiddleAbridgedText")),
    2000,
  );
  let end_len = h
    .get("history_end")
    .or_else(|| h.get("historyEnd"))
    .and_then(|x| x.as_array())
    .map(|a| a.len());
  let tmpl = truncate_inline_text(
    h.get("message_template")
      .or_else(|| h.get("messageTemplate")),
    400,
  );

  let mut lines: Vec<String> = vec!["[HISTORY_SUMMARY]".to_string()];
  if !req_id.is_empty() {
    lines.push(format!("summarization_request_id={req_id}"));
  }
  if let Some(n) = dropped {
    lines.push(format!("history_beginning_dropped_num_exchanges={n}"));
  }
  if !tmpl.is_empty() {
    lines.push(format!("message_template={tmpl}"));
  }
  if !summary_text.is_empty() {
    lines.push(format!("summary_text={summary_text}"));
  }
  if !abridged.is_empty() {
    lines.push(format!("history_middle_abridged_text={abridged}"));
  }
  if let Some(n) = end_len {
    if n > 0 {
      lines.push(format!("history_end_exchanges={n}"));
    }
  }
  if lines.len() == 1 {
    return String::new();
  }
  lines.push("[/HISTORY_SUMMARY]".to_string());
  lines.join("\n").trim().to_string()
}

#[derive(Debug, Default)]
pub struct AnthropicStreamState {
  pub node_id: i32,
  pub full_text: String,
  pub saw_tool_use: bool,
  pub stop_reason: Option<i32>,
  pub tool_meta_by_name: HashMap<String, (String, String)>,
  pub current_tool_use_id: Option<String>,
  pub current_tool_name: Option<String>,
  pub current_mcp_server_name: String,
  pub current_mcp_tool_name: String,
  pub tool_input_buffer: String,
  pub in_thinking_block: bool,
  pub thinking_buffer: String,
  pub usage_input_tokens: Option<i64>,
  pub usage_output_tokens: Option<i64>,
  pub usage_cache_read_input_tokens: Option<i64>,
  pub usage_cache_creation_input_tokens: Option<i64>,
  pub stop_reason_seen: bool,
}

impl AnthropicStreamState {
  pub fn on_text_delta(&mut self, delta: &str) -> crate::protocol::AugmentStreamChunk {
    self.full_text.push_str(delta);
    self.node_id += 1;
    crate::protocol::AugmentStreamChunk {
      text: delta.to_string(),
      nodes: vec![NodeOut {
        id: self.node_id,
        node_type: RESPONSE_NODE_RAW_RESPONSE,
        content: delta.to_string(),
        tool_use: None,
        thinking: None,
        token_usage: None,
      }],
      unknown_blob_names: Vec::new(),
      checkpoint_not_found: false,
      workspace_file_chunks: Vec::new(),
      stop_reason: None,
    }
  }

  pub fn on_usage(&mut self, usage: &crate::anthropic::AnthropicUsage) {
    if let Some(v) = usage.input_tokens {
      self.usage_input_tokens = Some(v);
    }
    if let Some(v) = usage.output_tokens {
      self.usage_output_tokens = Some(v);
    }
    if let Some(v) = usage.cache_read_input_tokens {
      self.usage_cache_read_input_tokens = Some(v);
    }
    if let Some(v) = usage.cache_creation_input_tokens {
      self.usage_cache_creation_input_tokens = Some(v);
    }
  }

  pub fn on_thinking_block_start(&mut self) {
    self.in_thinking_block = true;
    self.thinking_buffer.clear();
  }

  pub fn on_thinking_delta(&mut self, delta: &str) {
    self.thinking_buffer.push_str(delta);
  }

  pub fn on_thinking_block_stop(&mut self) -> Option<crate::protocol::AugmentStreamChunk> {
    if !self.in_thinking_block {
      return None;
    }
    self.in_thinking_block = false;
    if self.thinking_buffer.is_empty() {
      return None;
    }
    self.node_id += 1;
    let node = NodeOut {
      id: self.node_id,
      node_type: RESPONSE_NODE_THINKING,
      content: "".to_string(),
      tool_use: None,
      thinking: Some(ThinkingNode {
        summary: std::mem::take(&mut self.thinking_buffer),
      }),
      token_usage: None,
    };
    Some(crate::protocol::AugmentStreamChunk {
      text: "".to_string(),
      nodes: vec![node],
      unknown_blob_names: Vec::new(),
      checkpoint_not_found: false,
      workspace_file_chunks: Vec::new(),
      stop_reason: None,
    })
  }

  pub fn on_tool_use_block_start(&mut self, tool_use_id: &str, tool_name: &str) {
    let tool_name = tool_name.trim().to_string();
    self.current_tool_use_id = Some(tool_use_id.trim().to_string());
    self.current_tool_name = Some(tool_name.clone());
    self.tool_input_buffer.clear();
    if let Some((mcp_server_name, mcp_tool_name)) = self.tool_meta_by_name.get(&tool_name) {
      self.current_mcp_server_name = mcp_server_name.clone();
      self.current_mcp_tool_name = mcp_tool_name.clone();
    } else {
      self.current_mcp_server_name.clear();
      self.current_mcp_tool_name.clear();
    }
  }

  pub fn on_tool_input_json_delta(&mut self, partial: &str) {
    self.tool_input_buffer.push_str(partial);
  }

  pub fn on_tool_use_block_stop(&mut self) -> Vec<crate::protocol::AugmentStreamChunk> {
    let (Some(mut id), Some(name)) = (
      self.current_tool_use_id.take(),
      self.current_tool_name.take(),
    ) else {
      return Vec::new();
    };

    if id.trim().is_empty() {
      id = format!("tool-{}", self.node_id + 1);
    }
    let input_json = {
      let v = std::mem::take(&mut self.tool_input_buffer);
      let trimmed = v.trim();
      if trimmed.is_empty() {
        "{}".to_string()
      } else {
        trimmed.to_string()
      }
    };

    self.saw_tool_use = true;
    let mcp_server_name = std::mem::take(&mut self.current_mcp_server_name);
    let mcp_tool_name = std::mem::take(&mut self.current_mcp_tool_name);
    let tool_use = ToolUse {
      tool_use_id: id.clone(),
      tool_name: name,
      input_json,
      mcp_server_name,
      mcp_tool_name,
    };

    let mk_chunk = |node: NodeOut| crate::protocol::AugmentStreamChunk {
      text: "".to_string(),
      unknown_blob_names: Vec::new(),
      checkpoint_not_found: false,
      workspace_file_chunks: Vec::new(),
      nodes: vec![node],
      stop_reason: None,
    };

    self.node_id += 1;
    let start = mk_chunk(NodeOut {
      id: self.node_id,
      node_type: RESPONSE_NODE_TOOL_USE_START,
      content: "".to_string(),
      tool_use: Some(tool_use.clone()),
      thinking: None,
      token_usage: None,
    });

    self.node_id += 1;
    let done = mk_chunk(NodeOut {
      id: self.node_id,
      node_type: RESPONSE_NODE_TOOL_USE,
      content: "".to_string(),
      tool_use: Some(tool_use),
      thinking: None,
      token_usage: None,
    });

    vec![start, done]
  }

  pub fn on_stop_reason(&mut self, anthropic_stop_reason: &str) {
    self.stop_reason_seen = true;
    self.stop_reason = Some(map_anthropic_stop_reason_to_augment(anthropic_stop_reason));
  }

  pub fn finalize(&mut self) -> Vec<crate::protocol::AugmentStreamChunk> {
    let mut chunks: Vec<crate::protocol::AugmentStreamChunk> = Vec::new();

    if let Some(thinking) = self.on_thinking_block_stop() {
      chunks.push(thinking);
    }

    if self.usage_input_tokens.is_some()
      || self.usage_output_tokens.is_some()
      || self.usage_cache_read_input_tokens.is_some()
      || self.usage_cache_creation_input_tokens.is_some()
    {
      self.node_id += 1;
      chunks.push(crate::protocol::AugmentStreamChunk {
        text: "".to_string(),
        unknown_blob_names: Vec::new(),
        checkpoint_not_found: false,
        workspace_file_chunks: Vec::new(),
        nodes: vec![NodeOut {
          id: self.node_id,
          node_type: RESPONSE_NODE_TOKEN_USAGE,
          content: "".to_string(),
          tool_use: None,
          thinking: None,
          token_usage: Some(TokenUsageNode {
            input_tokens: self.usage_input_tokens,
            output_tokens: self.usage_output_tokens,
            cache_read_input_tokens: self.usage_cache_read_input_tokens,
            cache_creation_input_tokens: self.usage_cache_creation_input_tokens,
          }),
        }],
        stop_reason: None,
      });
    }

    let mut final_nodes: Vec<NodeOut> = Vec::new();
    if !self.full_text.is_empty() {
      self.node_id += 1;
      final_nodes.push(NodeOut {
        id: self.node_id,
        node_type: RESPONSE_NODE_MAIN_TEXT_FINISHED,
        content: self.full_text.clone(),
        tool_use: None,
        thinking: None,
        token_usage: None,
      });
    }

	    let stop_reason = self.stop_reason.unwrap_or_else(|| {
	      if self.saw_tool_use {
	        STOP_REASON_TOOL_USE_REQUESTED
	      } else {
	        STOP_REASON_END_TURN
	      }
	    });
    chunks.push(crate::protocol::AugmentStreamChunk {
      text: "".to_string(),
      unknown_blob_names: Vec::new(),
      checkpoint_not_found: false,
      workspace_file_chunks: Vec::new(),
      nodes: final_nodes,
      stop_reason: Some(stop_reason),
    });

    chunks
  }
}

#[derive(Debug, Default)]
pub struct OpenAIStreamState {
  pub node_id: i32,
  pub full_text: String,
  pub saw_tool_use: bool,
  pub stop_reason: Option<i32>,
  pub tool_meta_by_name: HashMap<String, (String, String)>,
  pub tool_calls: HashMap<usize, OpenAIToolCallBuffer>,
  pub usage_input_tokens: Option<i64>,
  pub usage_output_tokens: Option<i64>,
  pub stop_reason_seen: bool,
}

#[derive(Debug, Default)]
pub struct OpenAIToolCallBuffer {
  pub id: String,
  pub name: String,
  pub arguments: String,
  pub mcp_server_name: String,
  pub mcp_tool_name: String,
}

impl OpenAIStreamState {
  pub fn on_text_delta(&mut self, delta: &str) -> crate::protocol::AugmentStreamChunk {
    self.full_text.push_str(delta);
    self.node_id += 1;
    crate::protocol::AugmentStreamChunk {
      text: delta.to_string(),
      nodes: vec![NodeOut {
        id: self.node_id,
        node_type: RESPONSE_NODE_RAW_RESPONSE,
        content: delta.to_string(),
        tool_use: None,
        thinking: None,
        token_usage: None,
      }],
      unknown_blob_names: Vec::new(),
      checkpoint_not_found: false,
      workspace_file_chunks: Vec::new(),
      stop_reason: None,
    }
  }

  pub fn on_usage(&mut self, prompt_tokens: Option<i64>, completion_tokens: Option<i64>) {
    if let Some(v) = prompt_tokens {
      self.usage_input_tokens = Some(v);
    }
    if let Some(v) = completion_tokens {
      self.usage_output_tokens = Some(v);
    }
  }

  pub fn on_tool_call_delta(
    &mut self,
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
  ) {
    let entry = self
      .tool_calls
      .entry(index)
      .or_insert_with(OpenAIToolCallBuffer::default);

    if entry.id.trim().is_empty() {
      if let Some(id) = id.map(str::trim).filter(|s| !s.is_empty()) {
        entry.id = id.to_string();
      }
    }

    if entry.name.trim().is_empty() {
      if let Some(name) = name.map(str::trim).filter(|s| !s.is_empty()) {
        entry.name = name.to_string();
        if let Some((mcp_server_name, mcp_tool_name)) = self.tool_meta_by_name.get(name) {
          entry.mcp_server_name = mcp_server_name.clone();
          entry.mcp_tool_name = mcp_tool_name.clone();
        }
      }
    }

    if let Some(args) = arguments {
      entry.arguments.push_str(args);
    }
  }

  pub fn on_finish_reason(&mut self, finish_reason: &str) {
    self.stop_reason_seen = true;
    self.stop_reason = Some(map_openai_finish_reason_to_augment(finish_reason));
  }

  pub fn finalize(&mut self) -> Vec<crate::protocol::AugmentStreamChunk> {
    let mut chunks: Vec<crate::protocol::AugmentStreamChunk> = Vec::new();

    let mut indices: Vec<usize> = self.tool_calls.keys().copied().collect();
    indices.sort();
    for idx in indices {
      let Some(call) = self.tool_calls.get(&idx) else {
        continue;
      };
      let name = call.name.trim();
      if name.is_empty() {
        continue;
      }
      let mut id = call.id.trim().to_string();
      if id.is_empty() {
        id = format!("tool-{}", self.node_id + 1);
      }
      let input_json = {
        let v = call.arguments.trim();
        if v.is_empty() {
          "{}".to_string()
        } else {
          v.to_string()
        }
      };

      self.saw_tool_use = true;
      let tool_use = ToolUse {
        tool_use_id: id.clone(),
        tool_name: name.to_string(),
        input_json,
        mcp_server_name: call.mcp_server_name.clone(),
        mcp_tool_name: call.mcp_tool_name.clone(),
      };

      let mk_chunk = |node: NodeOut| crate::protocol::AugmentStreamChunk {
        text: "".to_string(),
        unknown_blob_names: Vec::new(),
        checkpoint_not_found: false,
        workspace_file_chunks: Vec::new(),
        nodes: vec![node],
        stop_reason: None,
      };

      self.node_id += 1;
      chunks.push(mk_chunk(NodeOut {
        id: self.node_id,
        node_type: RESPONSE_NODE_TOOL_USE_START,
        content: "".to_string(),
        tool_use: Some(tool_use.clone()),
        thinking: None,
        token_usage: None,
      }));

      self.node_id += 1;
      chunks.push(mk_chunk(NodeOut {
        id: self.node_id,
        node_type: RESPONSE_NODE_TOOL_USE,
        content: "".to_string(),
        tool_use: Some(tool_use),
        thinking: None,
        token_usage: None,
      }));
    }

    if self.usage_input_tokens.is_some() || self.usage_output_tokens.is_some() {
      self.node_id += 1;
      chunks.push(crate::protocol::AugmentStreamChunk {
        text: "".to_string(),
        unknown_blob_names: Vec::new(),
        checkpoint_not_found: false,
        workspace_file_chunks: Vec::new(),
        nodes: vec![NodeOut {
          id: self.node_id,
          node_type: RESPONSE_NODE_TOKEN_USAGE,
          content: "".to_string(),
          tool_use: None,
          thinking: None,
          token_usage: Some(TokenUsageNode {
            input_tokens: self.usage_input_tokens,
            output_tokens: self.usage_output_tokens,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
          }),
        }],
        stop_reason: None,
      });
    }

    let mut final_nodes: Vec<NodeOut> = Vec::new();
    if !self.full_text.is_empty() {
      self.node_id += 1;
      final_nodes.push(NodeOut {
        id: self.node_id,
        node_type: RESPONSE_NODE_MAIN_TEXT_FINISHED,
        content: self.full_text.clone(),
        tool_use: None,
        thinking: None,
        token_usage: None,
      });
    }

	    let stop_reason = self.stop_reason.unwrap_or_else(|| {
	      if self.saw_tool_use {
	        STOP_REASON_TOOL_USE_REQUESTED
	      } else {
	        STOP_REASON_END_TURN
	      }
	    });
    chunks.push(crate::protocol::AugmentStreamChunk {
      text: "".to_string(),
      unknown_blob_names: Vec::new(),
      checkpoint_not_found: false,
      workspace_file_chunks: Vec::new(),
      nodes: final_nodes,
      stop_reason: Some(stop_reason),
    });
    chunks
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::{AnthropicProviderConfig, OpenAICompatibleProviderConfig, ThinkingConfig};
  use crate::protocol::{
    AugmentChatHistory, AugmentRequest, NodeIn, TextNode, ToolDefinition, REQUEST_NODE_TEXT,
    RESPONSE_NODE_MAIN_TEXT_FINISHED, RESPONSE_NODE_RAW_RESPONSE, RESPONSE_NODE_THINKING,
    RESPONSE_NODE_TOOL_USE, RESPONSE_NODE_TOOL_USE_START,
  };
  use pretty_assertions::assert_eq;
  use std::collections::BTreeMap;

  fn make_text_node(id: i32, content: &str) -> NodeIn {
    NodeIn {
      id,
      node_type: REQUEST_NODE_TEXT,
      content: "".to_string(),
      text_node: Some(TextNode {
        content: content.to_string(),
      }),
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

  #[test]
  fn clean_model_handles_gemini_prefix() {
    assert_eq!(
      clean_model("gemini-claude-opus-4-5-thinking"),
      "claude-opus-4-5-thinking"
    );
    assert_eq!(
      clean_model("claude-opus-4-5-thinking"),
      "claude-opus-4-5-thinking"
    );
  }

  #[test]
  fn placeholder_and_history_request_nodes_are_handled() {
    assert_eq!(is_user_placeholder_message("-"), true);
    assert_eq!(is_user_placeholder_message("---"), true);
    assert_eq!(is_user_placeholder_message("----"), true);
    assert_eq!(is_user_placeholder_message("-----------------"), false);

    let blocks = build_user_content_blocks("---", [].iter(), true).unwrap();
    assert_eq!(blocks.len(), 0);

    let nodes = vec![make_text_node(1, "hi")];
    let blocks = build_user_content_blocks("---", nodes.iter(), true).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text.as_deref(), Some("hi"));

    let provider = AnthropicProviderConfig {
      id: "p1".to_string(),
      base_url: "https://api.anthropic.com/v1".to_string(),
      api_key: "sk-ant-dummy".to_string(),
      default_model: "claude-sonnet-4-20250514".to_string(),
      max_tokens: 8192,
      timeout_seconds: 120,
      thinking: ThinkingConfig {
        enabled: false,
        budget_tokens: 0,
      },
      extra_headers: BTreeMap::new(),
    };

    let augment = AugmentRequest {
      model: None,
      chat_history: vec![AugmentChatHistory {
        response_text: "ok".to_string(),
        request_message: "---".to_string(),
        request_id: "r1".to_string(),
        request_nodes: vec![make_text_node(1, "hi")],
        structured_request_nodes: Vec::new(),
        nodes: Vec::new(),
        response_nodes: Vec::new(),
        structured_output_nodes: Vec::new(),
      }],
      message: "".to_string(),
      agent_memories: "".to_string(),
      mode: "CHAT".to_string(),
      prefix: "".to_string(),
      suffix: "".to_string(),
      lang: "".to_string(),
      path: "".to_string(),
      user_guidelines: "".to_string(),
      tool_definitions: Vec::new(),
      nodes: Vec::new(),
      structured_request_nodes: Vec::new(),
      request_nodes: Vec::new(),
      conversation_id: None,
    };

    let out = convert_augment_to_anthropic(&provider, &augment, "m".to_string()).unwrap();
    assert_eq!(out.messages.len(), 2);
    assert_eq!(out.messages[0].role, "user");
    assert_eq!(out.messages[0].content.len(), 1);
    assert_eq!(out.messages[0].content[0].text.as_deref(), Some("hi"));
    assert_eq!(out.messages[1].role, "assistant");
    assert_eq!(out.messages[1].content.len(), 1);
    assert_eq!(out.messages[1].content[0].text.as_deref(), Some("ok"));
  }

  #[test]
  fn convert_tools_and_thinking_enabled() {
    let cfg = AnthropicProviderConfig {
      id: "p1".to_string(),
      base_url: "https://api.anthropic.com/v1".to_string(),
      api_key: "sk-ant-dummy".to_string(),
      default_model: "claude-sonnet-4-20250514".to_string(),
      max_tokens: 8192,
      timeout_seconds: 120,
      thinking: ThinkingConfig {
        enabled: true,
        budget_tokens: 10000,
      },
      extra_headers: BTreeMap::new(),
    };

    let augment = AugmentRequest {
      model: None,
      chat_history: Vec::new(),
      message: "hi".to_string(),
      agent_memories: "".to_string(),
      mode: "CHAT".to_string(),
      prefix: "P".to_string(),
      suffix: "".to_string(),
      lang: "rust".to_string(),
      path: "/tmp/main.rs".to_string(),
      user_guidelines: "G".to_string(),
      tool_definitions: vec![ToolDefinition {
        name: "t1".to_string(),
        description: "d1".to_string(),
        input_schema: None,
        input_schema_json: r#"{"type":"object","properties":{"q":{"type":"string"}}}"#.to_string(),
        tool_safety: None,
        mcp_server_name: "".to_string(),
        mcp_tool_name: "".to_string(),
      }],
      nodes: Vec::new(),
      structured_request_nodes: Vec::new(),
      request_nodes: Vec::new(),
      conversation_id: None,
    };

    let out = convert_augment_to_anthropic(&cfg, &augment, "m".to_string()).unwrap();
    assert_eq!(out.model, "m");
    assert_eq!(out.stream, true);
    assert_eq!(out.max_tokens, 8192);
    assert_eq!(out.tools.as_ref().unwrap().len(), 1);
    assert_eq!(out.tool_choice.as_ref().unwrap().choice_type, "auto");
    assert_eq!(out.thinking.as_ref().unwrap().thinking_type, "enabled");
    assert_eq!(out.thinking.as_ref().unwrap().budget_tokens, 10000);
    assert_eq!(out.system.as_ref().unwrap().contains("P"), true);
    assert_eq!(out.system.as_ref().unwrap().contains("G"), true);
    assert_eq!(out.system.as_ref().unwrap().contains("rust"), true);
    assert_eq!(out.system.as_ref().unwrap().contains("/tmp/main.rs"), true);
  }

  #[test]
  fn anthropic_stream_state_buffers_thinking_and_tool_use() {
    let mut s = AnthropicStreamState::default();

    let c1 = s.on_text_delta("hello");
    assert_eq!(c1.text, "hello");
    assert_eq!(c1.nodes.len(), 1);
    assert_eq!(c1.nodes[0].node_type, RESPONSE_NODE_RAW_RESPONSE);
    assert_eq!(c1.nodes[0].content, "hello");

    s.on_thinking_block_start();
    s.on_thinking_delta("t1");
    s.on_thinking_delta("t2");
    let thinking = s.on_thinking_block_stop().unwrap();
    assert_eq!(thinking.text, "");
    assert_eq!(thinking.nodes.len(), 1);
    assert_eq!(thinking.nodes[0].node_type, RESPONSE_NODE_THINKING);
    assert_eq!(
      thinking.nodes[0].thinking.as_ref().unwrap().summary,
      "t1t2".to_string()
    );

    s.on_tool_use_block_start("u1", "tool_a");
    s.on_tool_input_json_delta("{\"x\":");
    s.on_tool_input_json_delta("1}");
    let tool_chunks = s.on_tool_use_block_stop();
    assert_eq!(tool_chunks.len(), 2);
    assert_eq!(tool_chunks[0].stop_reason, None);
    assert_eq!(
      tool_chunks[0].nodes[0].node_type,
      RESPONSE_NODE_TOOL_USE_START
    );
    assert_eq!(
      tool_chunks[0].nodes[0]
        .tool_use
        .as_ref()
        .unwrap()
        .input_json,
      "{\"x\":1}".to_string()
    );
    assert_eq!(tool_chunks[1].stop_reason, None);
    assert_eq!(tool_chunks[1].nodes[0].node_type, RESPONSE_NODE_TOOL_USE);

    s.on_stop_reason("tool_use");
    let finals = s.finalize();
    assert_eq!(finals.len(), 1);
    assert_eq!(finals[0].stop_reason, Some(STOP_REASON_TOOL_USE_REQUESTED));
    assert_eq!(finals[0].nodes.len(), 1);
    assert_eq!(
      finals[0].nodes[0].node_type,
      RESPONSE_NODE_MAIN_TEXT_FINISHED
    );
    assert_eq!(finals[0].nodes[0].content, "hello");
  }

  #[test]
  fn convert_openai_includes_tools_and_stream_options() {
    let provider = OpenAICompatibleProviderConfig {
      id: "o1".to_string(),
      base_url: "https://api.openai.com/v1".to_string(),
      api_key: "sk-test".to_string(),
      default_model: "gpt-4o-mini".to_string(),
      max_tokens: 1234,
      timeout_seconds: 120,
      extra_headers: BTreeMap::new(),
    };

    let augment = AugmentRequest {
      model: None,
      chat_history: Vec::new(),
      message: "hi".to_string(),
      agent_memories: "".to_string(),
      mode: "CHAT".to_string(),
      prefix: "P".to_string(),
      suffix: "".to_string(),
      lang: "ts".to_string(),
      path: "/tmp/app.ts".to_string(),
      user_guidelines: "G".to_string(),
      tool_definitions: vec![ToolDefinition {
        name: "t1".to_string(),
        description: "d1".to_string(),
        input_schema: None,
        input_schema_json: r#"{"type":"object","properties":{"q":{"type":"string"}}}"#.to_string(),
        tool_safety: None,
        mcp_server_name: "".to_string(),
        mcp_tool_name: "".to_string(),
      }],
      nodes: Vec::new(),
      structured_request_nodes: Vec::new(),
      request_nodes: Vec::new(),
      conversation_id: None,
    };

    let out =
      convert_augment_to_openai_compatible(&provider, &augment, "gpt-4o-mini".to_string()).unwrap();
    assert_eq!(out.model, "gpt-4o-mini");
    assert_eq!(out.stream, true);
    assert_eq!(out.stream_options.as_ref().unwrap().include_usage, true);
    assert_eq!(out.max_tokens, Some(1234));
    assert_eq!(out.tools.as_ref().unwrap().len(), 1);
    assert_eq!(out.tool_choice.as_ref().unwrap().as_str().unwrap(), "auto");
    assert_eq!(out.messages[0].role, "system");
    assert_eq!(
      out.messages[0]
        .content
        .as_ref()
        .unwrap()
        .as_str()
        .unwrap()
        .contains("P"),
      true
    );
  }

  #[test]
  fn openai_stream_state_buffers_tool_calls_and_finalizes() {
    let mut state = OpenAIStreamState::default();
    state
      .tool_meta_by_name
      .insert("t1".to_string(), ("mcp".to_string(), "tool".to_string()));
    state.on_tool_call_delta(0, Some("call_1"), Some("t1"), Some("{\"q\":"));
    state.on_tool_call_delta(0, None, None, Some("\"hi\"}"));
    state.on_finish_reason("tool_calls");

    let chunks = state.finalize();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].nodes[0].node_type, RESPONSE_NODE_TOOL_USE_START);
    assert_eq!(chunks[1].nodes[0].node_type, RESPONSE_NODE_TOOL_USE);
    assert_eq!(
      chunks[1].nodes[0].tool_use.as_ref().unwrap().tool_use_id,
      "call_1"
    );
    assert_eq!(
      chunks[1].nodes[0].tool_use.as_ref().unwrap().tool_name,
      "t1"
    );
    assert_eq!(
      chunks[1].nodes[0].tool_use.as_ref().unwrap().input_json,
      "{\"q\":\"hi\"}"
    );
    assert_eq!(
      chunks[1].nodes[0]
        .tool_use
        .as_ref()
        .unwrap()
        .mcp_server_name,
      "mcp"
    );
    assert_eq!(
      chunks[1].nodes[0].tool_use.as_ref().unwrap().mcp_tool_name,
      "tool"
    );
    assert_eq!(chunks[2].stop_reason, Some(STOP_REASON_TOOL_USE_REQUESTED));
  }
}
