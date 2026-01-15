#![allow(dead_code)]

use serde::{Deserialize, Deserializer, Serialize};

pub fn de_null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
  D: Deserializer<'de>,
  T: Deserialize<'de> + Default,
{
  Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

pub const REQUEST_NODE_TEXT: i32 = 0;
pub const REQUEST_NODE_TOOL_RESULT: i32 = 1;
pub const REQUEST_NODE_IMAGE: i32 = 2;
pub const REQUEST_NODE_IMAGE_ID: i32 = 3;
pub const REQUEST_NODE_IDE_STATE: i32 = 4;
pub const REQUEST_NODE_EDIT_EVENTS: i32 = 5;
pub const REQUEST_NODE_CHECKPOINT_REF: i32 = 6;
pub const REQUEST_NODE_CHANGE_PERSONALITY: i32 = 7;
pub const REQUEST_NODE_FILE: i32 = 8;
pub const REQUEST_NODE_FILE_ID: i32 = 9;
pub const REQUEST_NODE_HISTORY_SUMMARY: i32 = 10;

pub const RESPONSE_NODE_RAW_RESPONSE: i32 = 0;
pub const RESPONSE_NODE_SUGGESTED_QUESTIONS: i32 = 1;
pub const RESPONSE_NODE_MAIN_TEXT_FINISHED: i32 = 2;
pub const RESPONSE_NODE_TOOL_USE: i32 = 5;
pub const RESPONSE_NODE_AGENT_MEMORY: i32 = 6;
pub const RESPONSE_NODE_TOOL_USE_START: i32 = 7;
pub const RESPONSE_NODE_THINKING: i32 = 8;
pub const RESPONSE_NODE_BILLING_METADATA: i32 = 9;
pub const RESPONSE_NODE_TOKEN_USAGE: i32 = 10;

// stop_reason 枚举需与官方扩展对齐（见 dist/_vsix_check/extension/common-webviews/assets/types-*.js）。
pub const STOP_REASON_UNSPECIFIED: i32 = 0;
pub const STOP_REASON_END_TURN: i32 = 1;
pub const STOP_REASON_MAX_TOKENS: i32 = 2;
pub const STOP_REASON_TOOL_USE_REQUESTED: i32 = 3;
pub const STOP_REASON_SAFETY: i32 = 4;
pub const STOP_REASON_RECITATION: i32 = 5;
pub const STOP_REASON_MALFORMED_FUNCTION_CALL: i32 = 6;

pub fn is_false(v: &bool) -> bool {
  !*v
}

#[derive(Debug, Deserialize)]
pub struct AugmentRequest {
  #[serde(default)]
  pub model: Option<String>,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "chatHistory"
  )]
  pub chat_history: Vec<AugmentChatHistory>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub message: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "agentMemories")]
  pub agent_memories: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub mode: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub prefix: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub suffix: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub lang: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub path: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "userGuidelines")]
  pub user_guidelines: String,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "toolDefinitions"
  )]
  pub tool_definitions: Vec<ToolDefinition>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub nodes: Vec<NodeIn>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "structuredRequestNodes")]
  pub structured_request_nodes: Vec<NodeIn>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "requestNodes")]
  pub request_nodes: Vec<NodeIn>,
  #[serde(default)]
  #[serde(alias = "conversationId")]
  pub conversation_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AugmentChatHistory {
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "responseText", alias = "response", alias = "text")]
  pub response_text: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "requestMessage", alias = "message")]
  pub request_message: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "requestId")]
  pub request_id: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "requestNodes")]
  pub request_nodes: Vec<NodeIn>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "structuredRequestNodes")]
  pub structured_request_nodes: Vec<NodeIn>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub nodes: Vec<NodeIn>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "responseNodes")]
  pub response_nodes: Vec<NodeIn>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "structuredOutputNodes")]
  pub structured_output_nodes: Vec<NodeIn>,
}

#[derive(Debug, Deserialize)]
pub struct ToolDefinition {
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub name: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub description: String,
  #[serde(default)]
  #[serde(alias = "inputSchema")]
  pub input_schema: Option<serde_json::Value>,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "inputSchemaJson")]
  pub input_schema_json: String,
  #[serde(default)]
  pub tool_safety: Option<i32>,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "mcpServerName"
  )]
  pub mcp_server_name: String,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "mcpToolName"
  )]
  pub mcp_tool_name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NodeIn {
  pub id: i32,
  #[serde(rename = "type")]
  pub node_type: i32,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub content: String,
  #[serde(default)]
  #[serde(alias = "textNode")]
  pub text_node: Option<TextNode>,
  #[serde(default)]
  #[serde(alias = "toolResultNode")]
  pub tool_result_node: Option<ToolResultNode>,
  #[serde(default)]
  #[serde(alias = "imageNode")]
  pub image_node: Option<ImageNode>,
  #[serde(default)]
  #[serde(alias = "imageIdNode")]
  pub image_id_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "ideStateNode")]
  pub ide_state_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "editEventsNode")]
  pub edit_events_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "checkpointRefNode")]
  pub checkpoint_ref_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "changePersonalityNode")]
  pub change_personality_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "fileNode")]
  pub file_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "fileIdNode")]
  pub file_id_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "historySummaryNode")]
  pub history_summary_node: Option<serde_json::Value>,
  #[serde(default)]
  #[serde(alias = "toolUse")]
  pub tool_use: Option<ToolUse>,
  #[serde(default)]
  pub thinking: Option<ThinkingNode>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextNode {
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub content: String,
}

pub const TOOL_RESULT_CONTENT_NODE_TEXT: i32 = 1;
pub const TOOL_RESULT_CONTENT_NODE_IMAGE: i32 = 2;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResultContentNode {
  #[serde(rename = "type")]
  pub node_type: i32,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "textContent"
  )]
  pub text_content: String,
  #[serde(default, alias = "imageContent")]
  pub image_content: Option<ToolResultImageContent>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResultImageContent {
  #[serde(default, deserialize_with = "de_null_as_default", alias = "imageData")]
  pub image_data: String,
  #[serde(default)]
  pub format: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolResultNode {
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "toolUseId")]
  pub tool_use_id: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub content: String,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "contentNodes"
  )]
  pub content_nodes: Vec<ToolResultContentNode>,
  #[serde(default)]
  pub is_error: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageNode {
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "imageData")]
  pub image_data: String,
  pub format: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolUse {
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "toolUseId")]
  pub tool_use_id: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "toolName")]
  pub tool_name: String,
  #[serde(default, deserialize_with = "de_null_as_default")]
  #[serde(alias = "inputJson")]
  pub input_json: String,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "mcpServerName",
    skip_serializing_if = "String::is_empty"
  )]
  pub mcp_server_name: String,
  #[serde(
    default,
    deserialize_with = "de_null_as_default",
    alias = "mcpToolName",
    skip_serializing_if = "String::is_empty"
  )]
  pub mcp_tool_name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThinkingNode {
  #[serde(default, deserialize_with = "de_null_as_default")]
  pub summary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenUsageNode {
  #[serde(default)]
  pub input_tokens: Option<i64>,
  #[serde(default)]
  pub output_tokens: Option<i64>,
  #[serde(default)]
  pub cache_read_input_tokens: Option<i64>,
  #[serde(default)]
  pub cache_creation_input_tokens: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct AugmentStreamChunk {
  pub text: String,
  #[serde(skip_serializing_if = "Vec::is_empty", default)]
  pub unknown_blob_names: Vec<String>,
  #[serde(skip_serializing_if = "is_false", default)]
  pub checkpoint_not_found: bool,
  #[serde(skip_serializing_if = "Vec::is_empty", default)]
  pub workspace_file_chunks: Vec<serde_json::Value>,
  #[serde(skip_serializing_if = "Vec::is_empty", default)]
  pub nodes: Vec<NodeOut>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub stop_reason: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct NodeOut {
  pub id: i32,
  #[serde(rename = "type")]
  pub node_type: i32,
  pub content: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tool_use: Option<ToolUse>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub thinking: Option<ThinkingNode>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub token_usage: Option<TokenUsageNode>,
}

pub fn probe_response() -> AugmentStreamChunk {
  AugmentStreamChunk {
    text: "".to_string(),
    unknown_blob_names: Vec::new(),
    checkpoint_not_found: false,
    workspace_file_chunks: Vec::new(),
    nodes: Vec::new(),
    stop_reason: Some(STOP_REASON_END_TURN),
  }
}

pub fn error_response(message: impl Into<String>) -> AugmentStreamChunk {
  AugmentStreamChunk {
    text: message.into(),
    unknown_blob_names: Vec::new(),
    checkpoint_not_found: false,
    workspace_file_chunks: Vec::new(),
    nodes: Vec::new(),
    stop_reason: Some(STOP_REASON_END_TURN),
  }
}
