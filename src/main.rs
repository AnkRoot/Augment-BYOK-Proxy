mod anthropic;
mod config;
mod convert;
mod history_summary;
mod history_summary_auto;
mod openai;
mod protocol;
mod util;

use std::{collections::HashMap, convert::Infallible, path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use async_stream::stream;
use axum::{
  body::{to_bytes, Body, Bytes},
  extract::{Query, State},
  http::{HeaderMap, HeaderValue, Request, Response, StatusCode},
  response::{Html, IntoResponse},
  routing::{get, post},
  Router,
};
use clap::Parser;
use futures::StreamExt;
use tokio::io::AsyncBufReadExt;
use tokio::sync::RwLock;
use tokio_util::io::StreamReader;
use tracing::{debug, error, info, warn};

use crate::{
  anthropic::AnthropicStreamEvent,
  config::{AnthropicProviderConfig, Config, OpenAICompatibleProviderConfig, ProviderConfig},
  convert::{
    clean_model, convert_augment_to_anthropic, convert_augment_to_openai_compatible,
    AnthropicStreamState, OpenAIStreamState,
  },
  history_summary::compact_chat_history,
  history_summary_auto::{maybe_summarize_and_compact, HistorySummaryCache},
  openai::OpenAIChatCompletionChunk,
  protocol::{error_response, probe_response, AugmentRequest, AugmentStreamChunk},
  util::{join_url, now_ms, normalize_raw_token},
};

#[derive(Debug, Clone, Copy)]
enum ProviderRef<'a> {
  Anthropic(&'a AnthropicProviderConfig),
  OpenAICompatible(&'a OpenAICompatibleProviderConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByokMode {
  Default,
  Official,
  Byok,
  Disabled,
}

impl<'a> ProviderRef<'a> {
  fn id(&self) -> &'a str {
    match self {
      ProviderRef::Anthropic(p) => p.id.as_str(),
      ProviderRef::OpenAICompatible(p) => p.id.as_str(),
    }
  }

  fn base_url(&self) -> &'a str {
    match self {
      ProviderRef::Anthropic(p) => p.base_url.as_str(),
      ProviderRef::OpenAICompatible(p) => p.base_url.as_str(),
    }
  }

  fn default_model(&self) -> &'a str {
    match self {
      ProviderRef::Anthropic(p) => p.default_model.as_str(),
      ProviderRef::OpenAICompatible(p) => p.default_model.as_str(),
    }
  }
}

#[derive(Debug, Parser)]
struct Args {
  #[arg(short, long, default_value = "config.yaml", value_name = "PATH")]
  config: PathBuf,
}

#[derive(Clone)]
struct AppState {
  config_path: PathBuf,
  cfg: Arc<RwLock<Config>>,
  http: reqwest::Client,
  models_cache: Arc<RwLock<ModelCache>>,
  history_summary_cache: Arc<RwLock<HistorySummaryCache>>,
}

#[derive(Debug, serde::Deserialize)]
struct ChatStreamQuery {
  #[serde(default)]
  model: Option<String>,
}

#[derive(Debug, Default)]
struct ModelCache {
  providers: HashMap<String, ModelCacheEntry>,
}

#[derive(Debug, Clone)]
struct ModelCacheEntry {
  kind: String,
  base_url: String,
  updated_at_ms: u64,
  models: Vec<String>,
}

const ADMIN_HTML: &str = r#"<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Augment-BYOK-Proxy Admin</title><style>body{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans","PingFang SC","Hiragino Sans GB","Microsoft YaHei";margin:24px;max-width:980px}textarea{width:100%;min-height:420px;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:12px}button{margin-right:8px;padding:8px 12px}small{color:#666}pre{background:#111;color:#eee;padding:12px;white-space:pre-wrap}</style></head><body><h1>Augment-BYOK-Proxy Admin</h1><p><small>说明：此页面直接编辑运行时配置（JSON）。<b>不支持</b>通过此接口热更新 <code>server.host/server.port</code> / <code>logging.filter</code>（需要重启进程）。</small></p><div style="margin:12px 0"><button id="reload">刷新</button><button id="apply">应用(热更新)</button><button id="save">保存到文件</button></div><textarea id="cfg" spellcheck="false"></textarea><h2>状态</h2><pre id="status">ready</pre><script>const $=s=>document.querySelector(s);const setStatus=(v)=>{$('#status').textContent=typeof v==='string'?v:JSON.stringify(v,null,2)};async function load(){setStatus('loading...');const r=await fetch('/admin/api/config');const t=await r.text();if(!r.ok){setStatus({ok:false,status:r.status,body:t});return}try{$('#cfg').value=JSON.stringify(JSON.parse(t),null,2);setStatus({ok:true})}catch(e){$('#cfg').value=t;setStatus({ok:false,error:'config JSON parse failed',detail:String(e)})}}async function apply(){let obj;try{obj=JSON.parse($('#cfg').value)}catch(e){setStatus({ok:false,error:'invalid JSON',detail:String(e)});return}setStatus('applying...');const r=await fetch('/admin/api/config',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(obj)});const j=await r.json().catch(async()=>({ok:false,body:await r.text()}));setStatus(j);if(r.ok)await load()}async function save(){setStatus('saving...');const r=await fetch('/admin/api/config/save',{method:'POST'});const j=await r.json().catch(async()=>({ok:false,body:await r.text()}));setStatus(j)}$('#reload').addEventListener('click',load);$('#apply').addEventListener('click',apply);$('#save').addEventListener('click',save);load();</script></body></html>"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  let args = Args::parse();
  let cfg = Config::load(&args.config)?;
  config::init_tracing(&cfg.logging)?;

  let addr = cfg.server.socket_addr()?;
  let http = reqwest::Client::builder().build()?;

  let state = AppState {
    config_path: args.config,
    cfg: Arc::new(RwLock::new(cfg)),
    http,
    models_cache: Arc::new(RwLock::new(ModelCache::default())),
    history_summary_cache: Arc::new(RwLock::new(HistorySummaryCache::default())),
  };

  let app = Router::new()
    .route("/health", get(health))
    .route("/chat", post(chat))
    .route("/completion", post(completion))
    .route("/chat-input-completion", post(chat_input_completion))
    .route("/edit", post(edit))
    .route("/prompt-enhancer", post(prompt_enhancer))
    .route("/instruction-stream", post(instruction_stream))
    .route("/smart-paste-stream", post(smart_paste_stream))
    .route(
      "/generate-commit-message-stream",
      post(generate_commit_message_stream),
    )
    .route("/generate-conversation-title", post(generate_conversation_title))
    .route("/chat-stream", post(chat_stream))
    .route("/get-models", post(get_models))
    .route("/admin", get(admin_index))
    .route(
      "/admin/api/config",
      get(admin_get_config).put(admin_put_config),
    )
    .route("/admin/api/config/save", post(admin_save_config))
    .fallback(proxy_fallback)
    .with_state(state)
    .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024));

  let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| {
    format!("监听 {addr} 失败（端口可能被占用；可修改 config.yaml 的 server.port）")
  })?;
  info!(%addr, "Augment-BYOK-Proxy 启动");
  axum::serve(listener, app).await?;
  Ok(())
}

async fn health() -> impl IntoResponse {
  axum::Json(serde_json::json!({ "status": "ok", "service": "augment-byok-proxy" }))
}

async fn admin_index() -> impl IntoResponse {
  Html(ADMIN_HTML)
}

async fn admin_get_config(State(state): State<AppState>) -> impl IntoResponse {
  let cfg = state.cfg.read().await.clone();
  axum::Json(cfg)
}

async fn admin_put_config(
  State(state): State<AppState>,
  axum::Json(mut next): axum::Json<Config>,
) -> impl IntoResponse {
  next.apply_defaults();
  if let Err(err) = next.validate() {
    return (
      StatusCode::BAD_REQUEST,
      axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") })),
    );
  }
  let current = state.cfg.read().await.clone();
  if next.server.host != current.server.host || next.server.port != current.server.port {
    return (
      StatusCode::BAD_REQUEST,
      axum::Json(
        serde_json::json!({ "ok": false, "error": "server.host/server.port 变更需要重启进程；本接口仅支持热更新上游相关配置" }),
      ),
    );
  }
  if next.logging.filter.trim() != current.logging.filter.trim() {
    return (
      StatusCode::BAD_REQUEST,
      axum::Json(
        serde_json::json!({ "ok": false, "error": "logging.filter 变更需要重启进程（暂不支持热更新）" }),
      ),
    );
  }
  *state.cfg.write().await = next;
  (
    StatusCode::OK,
    axum::Json(serde_json::json!({ "ok": true })),
  )
}

async fn admin_save_config(State(state): State<AppState>) -> impl IntoResponse {
  let cfg = state.cfg.read().await.clone();
  if let Err(err) = cfg.save(&state.config_path) {
    return (
      StatusCode::INTERNAL_SERVER_ERROR,
      axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") })),
    );
  }
  (
    StatusCode::OK,
    axum::Json(serde_json::json!({ "ok": true, "path": state.config_path.display().to_string() })),
  )
}

async fn chat_stream(
  State(state): State<AppState>,
  Query(query): Query<ChatStreamQuery>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&headers);
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    let present = auth_present_headers(&headers);
    warn!(present=?present, "chat-stream 未授权（缺少或错误的鉴权 token）");
    return ndjson_response(error_response(
      "❌ 未授权：请在 VS Code Settings 配置 augment.advanced.apiToken（需匹配 proxy.auth_token）",
    ));
  }
  if mode == ByokMode::Disabled {
    return ndjson_response(error_response("⛔ chat-stream 已被禁用（byok routing: disabled）"));
  }
  if mode == ByokMode::Official {
    let uri = if let Some(model) = query.model.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
      let mut ser = url::form_urlencoded::Serializer::new(String::new());
      ser.append_pair("model", model);
      let q = ser.finish();
      format!("/chat-stream?{q}")
        .parse::<axum::http::Uri>()
        .unwrap_or_else(|_| axum::http::Uri::from_static("/chat-stream"))
    } else {
      axum::http::Uri::from_static("/chat-stream")
    };
    let body_bytes = if body.is_empty() {
      Bytes::from_static(b"{}")
    } else {
      body
    };
    return forward_to_official(
      &state,
      &cfg,
      axum::http::Method::POST,
      &uri,
      &headers,
      body_bytes,
      Duration::from_secs(120),
    )
    .await;
  }
  let dump_body = cfg.logging.dump_chat_stream_body;
  if dump_body {
    info!(len=body.len(), body=%format_chat_stream_body_for_log(&body), "chat-stream 请求");
  } else {
    debug!(len = body.len(), "chat-stream 请求");
  }

  let mut augment = match parse_augment_request(&body) {
    Ok(v) => v,
    Err(err) => {
      error!(error=%err, len=body.len(), body=%format_chat_stream_body_for_log(&body), "chat-stream 请求解析失败");
      return ndjson_response(error_response(format!("⚠️ 请求解析失败: {err}")));
    }
  };

  let header_model = headers
    .get("x-byok-model")
    .or_else(|| headers.get("X-Byok-Model"))
    .or_else(|| headers.get("X-Model"))
    .or_else(|| headers.get("X-Augment-Model"))
    .or_else(|| headers.get("X-Anthropic-Model"))
    .or_else(|| headers.get("X-Custom-Model"))
    .and_then(|v| v.to_str().ok())
    .or_else(|| headers.get("Model").and_then(|v| v.to_str().ok()));

  let requested_model = query
    .model
    .as_deref()
    .filter(|s| !s.trim().is_empty())
    .or_else(|| header_model.filter(|s| !s.trim().is_empty()))
    .or_else(|| augment.model.as_deref().filter(|s| !s.trim().is_empty()))
    .unwrap_or("");

  let (provider, raw_model) = match parse_byok_model_id(requested_model) {
    Some((provider_id, model_id)) => match get_provider_by_id(&cfg, &provider_id) {
      Ok(p) => (p, model_id),
      Err(err) => return ndjson_response(error_response(format!("⚠️ 选择 provider 失败: {err}"))),
    },
    None => {
      let p = match pick_active_provider(&cfg) {
        Ok(v) => v,
        Err(err) => return ndjson_response(error_response(format!("⚠️ 缺少默认 provider: {err}"))),
      };
      let model = if requested_model.trim().is_empty() {
        p.default_model().to_string()
      } else {
        requested_model.to_string()
      };
      (p, model)
    }
  };

  let model_for_trigger = match provider {
    ProviderRef::Anthropic(_) => clean_model(&raw_model),
    ProviderRef::OpenAICompatible(_) => raw_model.trim().to_string(),
  };

  if let Err(err) = maybe_summarize_and_compact(
    &state.http,
    &cfg,
    &state.history_summary_cache,
    model_for_trigger.as_str(),
    &mut augment,
  )
  .await
  {
    warn!(error=%err, "history_summary 自动摘要失败（已忽略，继续使用原始 chat_history）");
  }
  compact_chat_history(&mut augment.chat_history);

  if augment.message.is_empty() && augment.chat_history.is_empty() {
    return ndjson_response(probe_response());
  }

  let tool_meta_by_name: std::collections::HashMap<String, (String, String)> = augment
    .tool_definitions
    .iter()
    .filter_map(|d| {
      let name = d.name.trim();
      if name.is_empty() {
        return None;
      }
      let mcp_server_name = d.mcp_server_name.trim();
      let mcp_tool_name = d.mcp_tool_name.trim();
      if mcp_server_name.is_empty() && mcp_tool_name.is_empty() {
        return None;
      }
      Some((
        name.to_string(),
        (mcp_server_name.to_string(), mcp_tool_name.to_string()),
      ))
    })
    .collect();

  match provider {
    ProviderRef::Anthropic(provider) => {
      let model = clean_model(&raw_model);
      let anthropic_req = match convert_augment_to_anthropic(provider, &augment, model) {
        Ok(v) => v,
        Err(err) => return ndjson_response(error_response(format!("⚠️ 转换请求失败: {err}"))),
      };

      let url = match join_url(&provider.base_url, "messages") {
        Ok(u) => u,
        Err(err) => {
          return ndjson_response(error_response(format!("⚠️ anthropic base_url 无效: {err}")))
        }
      };

      let api_key = normalize_raw_token(&provider.api_key);
      if api_key.is_empty() {
        return ndjson_response(error_response(format!(
          "⚠️ Provider({}) api_key 为空（请填写 byok.providers[].api_key；可用原始 token 或 KEY=VALUE 形式）",
          provider.id
        )));
      }

      let mut req = state
        .http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .header("anthropic-version", "2023-06-01")
        .header("x-api-key", api_key)
        .timeout(Duration::from_secs(provider.timeout_seconds))
        .json(&anthropic_req);

      for (k, v) in &provider.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          req = req.header(k, value);
        }
      }

      let resp = match req.send().await {
        Ok(r) => r,
        Err(err) => return ndjson_response(error_response(format!("❌ 上游请求失败: {err}"))),
      };

      if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return ndjson_response(error_response(format!(
          "❌ 上游返回错误: {status} {body_text}"
        )));
      }

      let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
      if dump_body {
        info!(status=%resp.status(), content_type=%content_type, "上游响应");
      } else {
        debug!(status=%resp.status(), content_type=%content_type, "上游响应");
      }
      if !content_type
        .trim()
        .to_ascii_lowercase()
        .contains("text/event-stream")
      {
        let body_text = resp.text().await.unwrap_or_default();
        let preview = truncate_for_log(body_text, 1024);
        return ndjson_response(error_response(format!(
          "❌ 上游响应不是 SSE（content-type={content_type}）；请确认 byok.providers[type=anthropic].base_url 指向 Anthropic Messages API 前缀（例如 https://api.anthropic.com/v1）；body: {preview}"
        )));
      }

      let tool_meta_by_name = tool_meta_by_name.clone();
      let stream = stream! {
        let mut state_machine = AnthropicStreamState::default();
        state_machine.tool_meta_by_name = tool_meta_by_name;
        let mut data_lines: usize = 0;
        let mut parsed_events: usize = 0;
        let mut emitted_chunks: usize = 0;
        let bytes_stream = resp.bytes_stream().map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
        let reader = StreamReader::new(bytes_stream);
        let mut lines = tokio::io::BufReader::new(reader).lines();
        let mut sse_event_type: Option<String> = None;

        while let Ok(Some(line)) = lines.next_line().await {
          if line.is_empty() {
            sse_event_type = None;
            continue;
          }
          if let Some(t) = line.strip_prefix("event:") {
            sse_event_type = Some(t.trim().to_string());
            continue;
          }
          let Some(data) = line.strip_prefix("data:") else { continue };
          let data = data.trim_start();
          data_lines += 1;

          let mut event: AnthropicStreamEvent = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
          };
          parsed_events += 1;
          if event.event_type.is_empty() {
            if let Some(t) = &sse_event_type {
              event.event_type = t.clone();
            }
          }

          for chunk in convert_event_to_chunks(&mut state_machine, event) {
            if let Ok(line) = serde_json::to_string(&chunk) {
              emitted_chunks += 1;
              yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
            }
          }
        }

        let has_usage = state_machine.usage_input_tokens.is_some()
          || state_machine.usage_output_tokens.is_some()
          || state_machine.usage_cache_read_input_tokens.is_some()
          || state_machine.usage_cache_creation_input_tokens.is_some();

        if emitted_chunks == 0 && !has_usage {
          let msg = format!("❌ 未解析到任何上游 SSE 内容（data_lines={data_lines}, parsed_events={parsed_events}）；请检查 byok.providers[type=anthropic].base_url 是否真的是 Anthropic /messages SSE");
          let error_chunk = error_response(msg);
          if let Ok(line) = serde_json::to_string(&error_chunk) {
            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
          }
          return;
        }

        for chunk in state_machine.finalize() {
          if let Ok(line) = serde_json::to_string(&chunk) {
            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
          }
        }
      };

      let mut response = Response::new(Body::from_stream(stream));
      let headers = response.headers_mut();
      headers.insert(
        "content-type",
        HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
      );
      headers.insert("cache-control", HeaderValue::from_static("no-cache"));
      headers.insert("connection", HeaderValue::from_static("keep-alive"));
      headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
      response
    }
    ProviderRef::OpenAICompatible(provider) => {
      let model = raw_model.trim().to_string();
      let openai_req = match convert_augment_to_openai_compatible(provider, &augment, model) {
        Ok(v) => v,
        Err(err) => return ndjson_response(error_response(format!("⚠️ 转换请求失败: {err}"))),
      };

      let url = match join_url(&provider.base_url, "chat/completions") {
        Ok(u) => u,
        Err(err) => {
          return ndjson_response(error_response(format!("⚠️ openai base_url 无效: {err}")))
        }
      };

      let api_key = normalize_raw_token(&provider.api_key);
      if api_key.is_empty() {
        return ndjson_response(error_response(format!(
          "⚠️ Provider({}) api_key 为空（请填写 byok.providers[].api_key；可用原始 token 或 KEY=VALUE 形式）",
          provider.id
        )));
      }

      let mut req = state
        .http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .header("authorization", format!("Bearer {api_key}"))
        .timeout(Duration::from_secs(provider.timeout_seconds))
        .json(&openai_req);

      for (k, v) in &provider.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          req = req.header(k, value);
        }
      }

      let resp = match req.send().await {
        Ok(r) => r,
        Err(err) => return ndjson_response(error_response(format!("❌ 上游请求失败: {err}"))),
      };

      if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return ndjson_response(error_response(format!(
          "❌ 上游返回错误: {status} {body_text}"
        )));
      }

      let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
      if dump_body {
        info!(status=%resp.status(), content_type=%content_type, "上游响应");
      } else {
        debug!(status=%resp.status(), content_type=%content_type, "上游响应");
      }
      if !content_type
        .trim()
        .to_ascii_lowercase()
        .contains("text/event-stream")
      {
        let body_text = resp.text().await.unwrap_or_default();
        let preview = truncate_for_log(body_text, 1024);
        return ndjson_response(error_response(format!(
          "❌ 上游响应不是 SSE（content-type={content_type}）；请确认 byok.providers[type=openai_compatible].base_url 指向 OpenAI Chat Completions API 前缀（例如 https://api.openai.com/v1）；body: {preview}"
        )));
      }

      let tool_meta_by_name = tool_meta_by_name.clone();
      let stream = stream! {
        let mut state_machine = OpenAIStreamState::default();
        state_machine.tool_meta_by_name = tool_meta_by_name;
        let mut data_lines: usize = 0;
        let mut parsed_chunks: usize = 0;
        let mut emitted_chunks: usize = 0;
        let bytes_stream = resp.bytes_stream().map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
        let reader = StreamReader::new(bytes_stream);
        let mut lines = tokio::io::BufReader::new(reader).lines();

        while let Ok(Some(line)) = lines.next_line().await {
          if line.is_empty() {
            continue;
          }
          let Some(data) = line.strip_prefix("data:") else { continue };
          let data = data.trim_start();
          data_lines += 1;
          if data == "[DONE]" {
            break;
          }

          let chunk: OpenAIChatCompletionChunk = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
          };
          parsed_chunks += 1;

          if let Some(u) = chunk.usage.as_ref() {
            state_machine.on_usage(u.prompt_tokens, u.completion_tokens);
          }

          for choice in &chunk.choices {
            if let Some(delta) = choice.delta.content.as_deref() {
              if !delta.is_empty() {
                let chunk = state_machine.on_text_delta(delta);
                if let Ok(line) = serde_json::to_string(&chunk) {
                  emitted_chunks += 1;
                  yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
                }
              }
            }

	            if let Some(calls) = choice.delta.tool_calls.as_ref() {
	              for c in calls {
	                let idx = c.index.unwrap_or(0);
	                let id = c.id.as_deref();
	                let name = c.function.as_ref().and_then(|f| f.name.as_deref());
	                let args = c.function.as_ref().and_then(|f| f.arguments.as_deref());
	                if let Some(chunk) = state_machine.on_tool_call_delta(idx, id, name, args) {
	                  if let Ok(line) = serde_json::to_string(&chunk) {
	                    emitted_chunks += 1;
	                    yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
	                  }
	                }
	              }
	            }

	            if let Some(fc) = choice.delta.function_call.as_ref() {
	              let name = fc.name.as_deref();
	              let args = fc.arguments.as_deref();
	              if let Some(chunk) = state_machine.on_tool_call_delta(0, None, name, args) {
	                if let Ok(line) = serde_json::to_string(&chunk) {
	                  emitted_chunks += 1;
	                  yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
	                }
	              }
	            }

            if let Some(r) = choice.finish_reason.as_deref() {
              state_machine.on_finish_reason(r);
            }
          }
        }

        let has_usage = state_machine.usage_input_tokens.is_some() || state_machine.usage_output_tokens.is_some();
        let has_tool_calls = !state_machine.tool_calls.is_empty();
        if emitted_chunks == 0 && !has_usage && !has_tool_calls {
          let msg = format!("❌ 未解析到任何上游 SSE 内容（data_lines={data_lines}, parsed_chunks={parsed_chunks}）；请检查 byok.providers[type=openai_compatible].base_url 是否真的是 OpenAI /chat/completions SSE");
          let error_chunk = error_response(msg);
          if let Ok(line) = serde_json::to_string(&error_chunk) {
            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
          }
          return;
        }

        for chunk in state_machine.finalize() {
          if let Ok(line) = serde_json::to_string(&chunk) {
            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
          }
        }
      };

      let mut response = Response::new(Body::from_stream(stream));
      let headers = response.headers_mut();
      headers.insert(
        "content-type",
        HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
      );
      headers.insert("cache-control", HeaderValue::from_static("no-cache"));
      headers.insert("connection", HeaderValue::from_static("keep-alive"));
      headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
      response
    }
  }
}

fn json_string(v: Option<&serde_json::Value>) -> String {
  match v {
    Some(serde_json::Value::String(s)) => s.trim().to_string(),
    Some(serde_json::Value::Null) => String::new(),
    Some(other) => other.as_str().unwrap_or("").trim().to_string(),
    None => String::new(),
  }
}

fn is_placeholder_message(message: &str) -> bool {
  let s = message.trim();
  if s.is_empty() {
    return false;
  }
  if !s.chars().all(|c| c == '-') {
    return false;
  }
  s.len() <= 16
}

fn build_system_text(body: &serde_json::Value) -> String {
  let mut parts: Vec<String> = Vec::new();

  fn push_lines(parts: &mut Vec<String>, label: &str, v: Option<&serde_json::Value>) {
    let Some(v) = v else { return };
    if let Some(s) = v.as_str().map(str::trim).filter(|s| !s.is_empty()) {
      parts.push(format!("{label}:\n{s}"));
      return;
    }
    if let Some(arr) = v.as_array() {
      let lines: Vec<String> = arr
        .iter()
        .filter_map(|x| x.as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string))
        .collect();
      if !lines.is_empty() {
        parts.push(format!("{label}:\n{}", lines.join("\n")));
      }
    }
  }

  push_lines(parts.as_mut(), "User Guidelines", body.get("user_guidelines"));
  push_lines(parts.as_mut(), "Workspace Guidelines", body.get("workspace_guidelines"));
  push_lines(parts.as_mut(), "Rules", body.get("rules"));

  parts.join("\n\n").trim().to_string()
}

fn build_user_text(body: &serde_json::Value) -> String {
  let has_nodes = body
    .get("nodes")
    .and_then(|v| v.as_array())
    .map(|arr| arr.iter().any(|x| x.is_object()))
    .unwrap_or(false);

  let message = json_string(body.get("message"));
  let prompt = json_string(body.get("prompt"));
  let instruction = json_string(body.get("instruction"));

  let use_message = !message.is_empty() && !is_placeholder_message(&message);
  let use_prompt = !use_message && !prompt.is_empty();
  let main = if use_message {
    message.clone()
  } else if use_prompt {
    prompt.clone()
  } else {
    instruction.clone()
  };

  let mut parts: Vec<String> = Vec::new();
  if !main.is_empty() {
    parts.push(main.clone());
  }

  if has_nodes || use_prompt {
    return parts.join("\n\n").trim().to_string();
  }

  let prefix = json_string(body.get("prefix"));
  let selected = json_string(body.get("selected_text").or_else(|| body.get("selected_code")));
  let suffix = json_string(body.get("suffix"));
  let code = format!("{prefix}{selected}{suffix}").trim().to_string();
  if !code.is_empty() && code != main.trim() {
    parts.push(code.clone());
  }
  let diff = json_string(body.get("diff"));
  if !diff.is_empty() && diff != code && diff != main.trim() {
    parts.push(diff);
  }
  parts.join("\n\n").trim().to_string()
}

fn read_model_from_body(body: &serde_json::Value) -> Option<String> {
  let keys = [
    "model",
    "model_id",
    "modelId",
    "provider_model_name",
    "providerModelName",
  ];
  for k in keys {
    let s = json_string(body.get(k));
    if !s.is_empty() {
      return Some(s);
    }
  }
  None
}

fn pick_provider_and_model_for_simple<'a>(
  cfg: &'a Config,
  headers: &HeaderMap,
  body: &serde_json::Value,
) -> anyhow::Result<(ProviderRef<'a>, String)> {
  let requested_model = read_byok_model_override(headers)
    .or_else(|| read_model_from_body(body))
    .unwrap_or_default();

  let (provider, raw_model) = match parse_byok_model_id(&requested_model) {
    Some((provider_id, model_id)) => match get_provider_by_id(cfg, &provider_id) {
      Ok(p) => (p, model_id),
      Err(err) => anyhow::bail!("选择 provider 失败: {err}"),
    },
    None => {
      let p = pick_active_provider(cfg)?;
      let model = if requested_model.trim().is_empty() {
        p.default_model().to_string()
      } else {
        requested_model
      };
      (p, model)
    }
  };

  let model = match provider {
    ProviderRef::Anthropic(_) => clean_model(&raw_model),
    ProviderRef::OpenAICompatible(_) => raw_model.trim().to_string(),
  };
  Ok((provider, model))
}

async fn provider_complete_text(
  state: &AppState,
  provider: ProviderRef<'_>,
  model: &str,
  system: &str,
  user: &str,
) -> anyhow::Result<String> {
  match provider {
    ProviderRef::Anthropic(p) => {
      let url = join_url(&p.base_url, "messages").context("构建 Anthropic messages URL 失败")?;
      let key = normalize_raw_token(&p.api_key);
      if key.is_empty() {
        anyhow::bail!("Provider({}) api_key 为空", p.id);
      }

      let mut payload = serde_json::json!({
        "model": model,
        "max_tokens": p.max_tokens,
        "stream": false,
        "messages": [{ "role": "user", "content": [{ "type": "text", "text": user }] }]
      });
      if !system.trim().is_empty() {
        if let Some(obj) = payload.as_object_mut() {
          obj.insert("system".to_string(), serde_json::Value::String(system.trim().to_string()));
        }
      }
      if p.thinking.enabled {
        if let Some(obj) = payload.as_object_mut() {
          obj.insert(
            "thinking".to_string(),
            serde_json::json!({ "type": "enabled", "budget_tokens": p.thinking.budget_tokens }),
          );
        }
      }

      let mut req = state
        .http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("x-api-key", key)
        .timeout(Duration::from_secs(p.timeout_seconds))
        .json(&payload);

      for (k, v) in &p.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          req = req.header(k, value);
        }
      }

      let resp = req.send().await.context("请求 Anthropic /messages 失败")?;
      let status = resp.status();
      let text = resp.text().await.unwrap_or_default();
      if !status.is_success() {
        anyhow::bail!("Anthropic /messages 返回错误: {status} {text}");
      }
      let json: serde_json::Value =
        serde_json::from_str(&text).context("Anthropic /messages 响应不是 JSON")?;
      let mut out = String::new();
      if let Some(arr) = json.get("content").and_then(|v| v.as_array()) {
        for b in arr {
          let t = b.get("type").and_then(|v| v.as_str()).unwrap_or("");
          if t == "text" {
            if let Some(s) = b.get("text").and_then(|v| v.as_str()) {
              out.push_str(s);
            }
          }
        }
      }
      Ok(out.trim().to_string())
    }
    ProviderRef::OpenAICompatible(p) => {
      let url =
        join_url(&p.base_url, "chat/completions").context("构建 OpenAI chat/completions URL 失败")?;
      let key = normalize_raw_token(&p.api_key);
      if key.is_empty() {
        anyhow::bail!("Provider({}) api_key 为空", p.id);
      }

      let mut messages: Vec<serde_json::Value> = Vec::new();
      if !system.trim().is_empty() {
        messages.push(serde_json::json!({ "role": "system", "content": system.trim() }));
      }
      messages.push(serde_json::json!({ "role": "user", "content": user }));

      let payload = serde_json::json!({
        "model": model,
        "stream": false,
        "max_tokens": p.max_tokens,
        "messages": messages
      });

      let mut req = state
        .http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .header("authorization", format!("Bearer {key}"))
        .timeout(Duration::from_secs(p.timeout_seconds))
        .json(&payload);

      for (k, v) in &p.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          req = req.header(k, value);
        }
      }

      let resp = req.send().await.context("请求 OpenAI /chat/completions 失败")?;
      let status = resp.status();
      let text = resp.text().await.unwrap_or_default();
      if !status.is_success() {
        anyhow::bail!("OpenAI /chat/completions 返回错误: {status} {text}");
      }
      let json: serde_json::Value =
        serde_json::from_str(&text).context("OpenAI /chat/completions 响应不是 JSON")?;
      let content = json
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
      Ok(content)
    }
  }
}

async fn byok_text_stream_endpoint(
  state: AppState,
  cfg: Config,
  headers: HeaderMap,
  body: Bytes,
  output_text_only: bool,
  endpoint_path: &'static str,
) -> Response<Body> {
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    let mut resp = Response::new(Body::from("Unauthorized"));
    *resp.status_mut() = StatusCode::UNAUTHORIZED;
    return resp;
  }
  let mode = read_byok_mode(&headers);
  if mode == ByokMode::Disabled {
    let mut resp = Response::new(Body::from("Disabled by routing rule"));
    *resp.status_mut() = StatusCode::NOT_FOUND;
    return resp;
  }
  if mode != ByokMode::Byok {
    let uri = axum::http::Uri::from_static(endpoint_path);
    return forward_to_official(
      &state,
      &cfg,
      axum::http::Method::POST,
      &uri,
      &headers,
      body,
      Duration::from_secs(120),
    )
    .await;
  }

  let value: serde_json::Value = match serde_json::from_slice(&body) {
    Ok(v) => v,
    Err(err) => {
      let mut resp = Response::new(Body::from(format!("Bad request JSON: {err}")));
      *resp.status_mut() = StatusCode::BAD_REQUEST;
      return resp;
    }
  };
  if value.get("encrypted_data").is_some() {
    let mut resp = Response::new(Body::from("encrypted_data not supported"));
    *resp.status_mut() = StatusCode::BAD_REQUEST;
    return resp;
  }

  let (provider, model) = match pick_provider_and_model_for_simple(&cfg, &headers, &value) {
    Ok(v) => v,
    Err(err) => {
      let mut resp = Response::new(Body::from(format!("Bad request: {err}")));
      *resp.status_mut() = StatusCode::BAD_REQUEST;
      return resp;
    }
  };

  let system = build_system_text(&value);
  let user = build_user_text(&value);

  let resp = match provider {
    ProviderRef::Anthropic(p) => {
      let url = match join_url(&p.base_url, "messages") {
        Ok(u) => u,
        Err(err) => {
          let mut resp = Response::new(Body::from(format!("Bad request: {err}")));
          *resp.status_mut() = StatusCode::BAD_REQUEST;
          return resp;
        }
      };
      let key = normalize_raw_token(&p.api_key);
      if key.is_empty() {
        let mut resp = Response::new(Body::from(format!("Provider({}) api_key 为空", p.id)));
        *resp.status_mut() = StatusCode::BAD_REQUEST;
        return resp;
      }

      let mut payload = serde_json::json!({
        "model": model,
        "max_tokens": p.max_tokens,
        "stream": true,
        "messages": [{ "role": "user", "content": [{ "type": "text", "text": user }] }]
      });
      if !system.trim().is_empty() {
        if let Some(obj) = payload.as_object_mut() {
          obj.insert("system".to_string(), serde_json::Value::String(system.trim().to_string()));
        }
      }
      if p.thinking.enabled {
        if let Some(obj) = payload.as_object_mut() {
          obj.insert(
            "thinking".to_string(),
            serde_json::json!({ "type": "enabled", "budget_tokens": p.thinking.budget_tokens }),
          );
        }
      }

      let mut req = state
        .http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .header("anthropic-version", "2023-06-01")
        .header("x-api-key", key)
        .timeout(Duration::from_secs(p.timeout_seconds))
        .json(&payload);

      for (k, v) in &p.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          req = req.header(k, value);
        }
      }
      req.send().await
    }
    ProviderRef::OpenAICompatible(p) => {
      let url = match join_url(&p.base_url, "chat/completions") {
        Ok(u) => u,
        Err(err) => {
          let mut resp = Response::new(Body::from(format!("Bad request: {err}")));
          *resp.status_mut() = StatusCode::BAD_REQUEST;
          return resp;
        }
      };
      let key = normalize_raw_token(&p.api_key);
      if key.is_empty() {
        let mut resp = Response::new(Body::from(format!("Provider({}) api_key 为空", p.id)));
        *resp.status_mut() = StatusCode::BAD_REQUEST;
        return resp;
      }

      let mut messages: Vec<serde_json::Value> = Vec::new();
      if !system.trim().is_empty() {
        messages.push(serde_json::json!({ "role": "system", "content": system.trim() }));
      }
      messages.push(serde_json::json!({ "role": "user", "content": user }));

      let payload = serde_json::json!({
        "model": model,
        "stream": true,
        "max_tokens": p.max_tokens,
        "messages": messages
      });

      let mut req = state
        .http
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .header("authorization", format!("Bearer {key}"))
        .timeout(Duration::from_secs(p.timeout_seconds))
        .json(&payload);

      for (k, v) in &p.extra_headers {
        if let Ok(value) = HeaderValue::from_str(v) {
          req = req.header(k, value);
        }
      }
      req.send().await
    }
  };

  let resp = match resp {
    Ok(r) => r,
    Err(err) => {
      let mut resp = Response::new(Body::from(format!("Upstream request failed: {err}")));
      *resp.status_mut() = StatusCode::BAD_GATEWAY;
      return resp;
    }
  };

  if !resp.status().is_success() {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    let mut out = Response::new(Body::from(format!("Upstream error: {status} {text}")));
    *out.status_mut() = StatusCode::BAD_GATEWAY;
    return out;
  }

  let stream = stream! {
    let bytes_stream = resp.bytes_stream().map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
    let reader = StreamReader::new(bytes_stream);
    let mut lines = tokio::io::BufReader::new(reader).lines();

    let mut anthropic_event_type: Option<String> = None;
    while let Ok(Some(line)) = lines.next_line().await {
      if line.is_empty() {
        anthropic_event_type = None;
        continue;
      }
      if line.starts_with("event:") {
        anthropic_event_type = Some(line.trim_start_matches("event:").trim().to_string());
        continue;
      }
      let Some(data) = line.strip_prefix("data:") else { continue };
      let data = data.trim_start();
      if data == "[DONE]" {
        break;
      }

      let mut text_delta: Option<String> = None;
      if let Ok(mut ev) = serde_json::from_str::<AnthropicStreamEvent>(data) {
        if ev.event_type.is_empty() {
          if let Some(t) = &anthropic_event_type {
            ev.event_type = t.clone();
          }
        }
        if ev.event_type == "content_block_delta" {
          if let Some(delta) = ev.delta {
            if delta.delta_type == "text_delta" {
              if let Some(t) = delta.text {
                if !t.is_empty() {
                  text_delta = Some(t);
                }
              }
            }
          }
        }
      } else if let Ok(chunk) = serde_json::from_str::<OpenAIChatCompletionChunk>(data) {
        for c in chunk.choices {
          if let Some(t) = c.delta.content {
            if !t.is_empty() {
              text_delta = Some(t);
              break;
            }
          }
        }
      }

      let Some(t) = text_delta else { continue };
      let raw = if output_text_only {
        serde_json::json!({ "text": t })
      } else {
        serde_json::json!({ "text": t, "unknown_blob_names": [], "checkpoint_not_found": false, "workspace_file_chunks": [], "nodes": [] })
      };
      if let Ok(line) = serde_json::to_string(&raw) {
        yield Ok::<Bytes, Infallible>(Bytes::from(format!("{line}\n")));
      }
    }
  };

  let mut response = Response::new(Body::from_stream(stream));
  response.headers_mut().insert(
    "content-type",
    HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
  );
  response
}

async fn chat(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&headers);
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({ "ok": false, "error": "Unauthorized" }))).into_response();
  }
  if mode == ByokMode::Disabled {
    return (StatusCode::NOT_FOUND, axum::Json(serde_json::json!({ "ok": false, "error": "Disabled by routing rule" }))).into_response();
  }
  if mode != ByokMode::Byok {
    let uri = axum::http::Uri::from_static("/chat");
    return forward_to_official(&state, &cfg, axum::http::Method::POST, &uri, &headers, body, Duration::from_secs(120)).await;
  }

  let value: serde_json::Value = match serde_json::from_slice(&body) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("Bad JSON: {err}") }))).into_response(),
  };
  if value.get("encrypted_data").is_some() {
    return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": "encrypted_data not supported" }))).into_response();
  }
  let (provider, model) = match pick_provider_and_model_for_simple(&cfg, &headers, &value) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let system = build_system_text(&value);
  let user = build_user_text(&value);
  let text = match provider_complete_text(&state, provider, &model, &system, &user).await {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  (StatusCode::OK, axum::Json(serde_json::json!({ "text": text, "unknown_blob_names": [], "checkpoint_not_found": false, "workspace_file_chunks": [], "nodes": [] }))).into_response()
}

async fn completion(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&headers);
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({ "ok": false, "error": "Unauthorized" }))).into_response();
  }
  if mode == ByokMode::Disabled {
    return (StatusCode::NOT_FOUND, axum::Json(serde_json::json!({ "ok": false, "error": "Disabled by routing rule" }))).into_response();
  }
  if mode != ByokMode::Byok {
    let uri = axum::http::Uri::from_static("/completion");
    return forward_to_official(&state, &cfg, axum::http::Method::POST, &uri, &headers, body, Duration::from_secs(120)).await;
  }

  let value: serde_json::Value = match serde_json::from_slice(&body) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("Bad JSON: {err}") }))).into_response(),
  };
  if value.get("encrypted_data").is_some() {
    return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": "encrypted_data not supported" }))).into_response();
  }
  let (provider, model) = match pick_provider_and_model_for_simple(&cfg, &headers, &value) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let system = build_system_text(&value);
  let user = build_user_text(&value);
  let text = match provider_complete_text(&state, provider, &model, &system, &user).await {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let out = serde_json::json!({
    "completion_items": [{ "text": text, "suffix_replacement_text": "", "skipped_suffix": "" }],
    "unknown_blob_names": [],
    "checkpoint_not_found": false,
    "suggested_prefix_char_count": 0,
    "suggested_suffix_char_count": 0,
    "completion_timeout_ms": 0
  });
  (StatusCode::OK, axum::Json(out)).into_response()
}

async fn chat_input_completion(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&headers);
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({ "ok": false, "error": "Unauthorized" }))).into_response();
  }
  if mode == ByokMode::Disabled {
    return (StatusCode::NOT_FOUND, axum::Json(serde_json::json!({ "ok": false, "error": "Disabled by routing rule" }))).into_response();
  }
  if mode != ByokMode::Byok {
    let uri = axum::http::Uri::from_static("/chat-input-completion");
    return forward_to_official(&state, &cfg, axum::http::Method::POST, &uri, &headers, body, Duration::from_secs(120)).await;
  }

  let value: serde_json::Value = match serde_json::from_slice(&body) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("Bad JSON: {err}") }))).into_response(),
  };
  if value.get("encrypted_data").is_some() {
    return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": "encrypted_data not supported" }))).into_response();
  }
  let (provider, model) = match pick_provider_and_model_for_simple(&cfg, &headers, &value) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let system = build_system_text(&value);
  let user = build_user_text(&value);
  let text = match provider_complete_text(&state, provider, &model, &system, &user).await {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let out = serde_json::json!({
    "completion_items": [{ "text": text, "suffix_replacement_text": "", "skipped_suffix": "" }],
    "unknown_blob_names": [],
    "checkpoint_not_found": false,
    "suggested_prefix_char_count": 0,
    "suggested_suffix_char_count": 0,
    "completion_timeout_ms": 0
  });
  (StatusCode::OK, axum::Json(out)).into_response()
}

async fn edit(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&headers);
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({ "ok": false, "error": "Unauthorized" }))).into_response();
  }
  if mode == ByokMode::Disabled {
    return (StatusCode::NOT_FOUND, axum::Json(serde_json::json!({ "ok": false, "error": "Disabled by routing rule" }))).into_response();
  }
  if mode != ByokMode::Byok {
    let uri = axum::http::Uri::from_static("/edit");
    return forward_to_official(&state, &cfg, axum::http::Method::POST, &uri, &headers, body, Duration::from_secs(120)).await;
  }

  let value: serde_json::Value = match serde_json::from_slice(&body) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("Bad JSON: {err}") }))).into_response(),
  };
  if value.get("encrypted_data").is_some() {
    return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": "encrypted_data not supported" }))).into_response();
  }
  let (provider, model) = match pick_provider_and_model_for_simple(&cfg, &headers, &value) {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let system = build_system_text(&value);
  let user = build_user_text(&value);
  let text = match provider_complete_text(&state, provider, &model, &system, &user).await {
    Ok(v) => v,
    Err(err) => return (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({ "ok": false, "error": format!("{err}") }))).into_response(),
  };
  let out = serde_json::json!({ "text": text, "unknown_blob_names": [], "checkpoint_not_found": false });
  (StatusCode::OK, axum::Json(out)).into_response()
}

async fn prompt_enhancer(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  byok_text_stream_endpoint(state, cfg, headers, body, false, "/prompt-enhancer").await
}

async fn instruction_stream(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  byok_text_stream_endpoint(state, cfg, headers, body, true, "/instruction-stream").await
}

async fn smart_paste_stream(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  byok_text_stream_endpoint(state, cfg, headers, body, true, "/smart-paste-stream").await
}

async fn generate_commit_message_stream(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  byok_text_stream_endpoint(
    state,
    cfg,
    headers,
    body,
    false,
    "/generate-commit-message-stream",
  )
  .await
}

async fn generate_conversation_title(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  byok_text_stream_endpoint(state, cfg, headers, body, false, "/generate-conversation-title").await
}

async fn get_models(
  State(state): State<AppState>,
  headers: HeaderMap,
  body: Bytes,
) -> Response<Body> {
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&headers);
  if !is_authorized(&headers, &cfg.proxy.auth_token) {
    let present = auth_present_headers(&headers);
    warn!(present=?present, "get-models 未授权（缺少或错误的鉴权 token）");
    return (
      StatusCode::UNAUTHORIZED,
      axum::Json(
        serde_json::json!({ "ok": false, "error": "Unauthorized: invalid apiToken (expected proxy.auth_token)" }),
      ),
    )
      .into_response();
  }

  if mode == ByokMode::Disabled {
    return (
      StatusCode::NOT_FOUND,
      axum::Json(serde_json::json!({ "ok": false, "error": "Disabled by routing rule" })),
    )
      .into_response();
  }
  if mode == ByokMode::Official {
    let uri = axum::http::Uri::from_static("/get-models");
    let body_bytes = if body.is_empty() {
      Bytes::from_static(b"{}")
    } else {
      body
    };
    return forward_to_official(
      &state,
      &cfg,
      axum::http::Method::POST,
      &uri,
      &headers,
      body_bytes,
      Duration::from_secs(12),
    )
    .await;
  }

  let url = match join_url(&cfg.official.base_url, "get-models") {
    Ok(u) => u,
    Err(err) => {
      return (
        StatusCode::BAD_REQUEST,
        axum::Json(
          serde_json::json!({ "ok": false, "error": format!("official.base_url 无效: {err}") }),
        ),
      )
        .into_response()
    }
  };

  let official_token = normalize_raw_token(&cfg.official.api_token);
  if official_token.is_empty() {
    return (
      StatusCode::BAD_REQUEST,
      axum::Json(serde_json::json!({ "ok": false, "error": "official.api_token 为空" })),
    )
      .into_response();
  }

  let mut req = state
    .http
    .post(url)
    .header("content-type", "application/json")
    .header("accept", "application/json")
    .header("authorization", format!("Bearer {official_token}"))
    .timeout(Duration::from_secs(12));

  for (k, v) in headers.iter() {
    if should_drop_forward_header(k.as_str()) {
      continue;
    }
    req = req.header(k, v);
  }

  let upstream_body = if body.is_empty() {
    Bytes::from_static(b"{}")
  } else {
    body
  };
  let resp = match req.body(upstream_body).send().await {
    Ok(r) => r,
    Err(err) => {
      return (
        StatusCode::BAD_GATEWAY,
        axum::Json(
          serde_json::json!({ "ok": false, "error": format!("上游 get-models 请求失败: {err}") }),
        ),
      )
        .into_response()
    }
  };

  let status = resp.status();
  let text = resp.text().await.unwrap_or_default();
  if !status.is_success() {
    return (
      StatusCode::BAD_GATEWAY,
      axum::Json(
        serde_json::json!({ "ok": false, "error": format!("上游 get-models 返回错误: {status} {text}") }),
      ),
    )
      .into_response();
  }

  let mut upstream: serde_json::Value = match serde_json::from_str(&text) {
    Ok(v) => v,
    Err(err) => {
      return (
        StatusCode::BAD_GATEWAY,
        axum::Json(
          serde_json::json!({ "ok": false, "error": format!("上游 get-models 响应不是 JSON: {err}") }),
        ),
      )
        .into_response()
    }
  };

  let Some(obj) = upstream.as_object_mut() else {
    return (
      StatusCode::BAD_GATEWAY,
      axum::Json(
        serde_json::json!({ "ok": false, "error": "上游 get-models 响应不是 JSON object" }),
      ),
    )
      .into_response();
  };

  let active = match pick_active_provider(&cfg) {
    Ok(v) => v,
    Err(err) => {
      return (
        StatusCode::BAD_REQUEST,
        axum::Json(
          serde_json::json!({ "ok": false, "error": format!("缺少 byok providers: {err}") }),
        ),
      )
        .into_response()
    }
  };

  let active_id = active.id().trim().to_string();
  let mut ordered: Vec<ProviderRef<'_>> = Vec::new();
  ordered.push(active);
  for p in &cfg.byok.providers {
    let p = match p {
      ProviderConfig::Anthropic(p) => ProviderRef::Anthropic(p),
      ProviderConfig::OpenAICompatible(p) => ProviderRef::OpenAICompatible(p),
    };
    if p.id().trim() == active_id {
      continue;
    }
    ordered.push(p);
  }

  let mut registry = serde_json::Map::<String, serde_json::Value>::new();
  let mut info_registry = serde_json::Map::<String, serde_json::Value>::new();
  let mut models: Vec<serde_json::Value> = Vec::new();

  for p in ordered {
    let mut list = Vec::<String>::new();
    if !p.default_model().trim().is_empty() {
      list.push(p.default_model().trim().to_string());
    }
    match get_provider_models_cached(&state, p).await {
      Ok(ms) => list.extend(ms),
      Err(err) => {
        warn!(provider=%p.id(), error=%err, "Provider models 拉取失败（将仅保留 default_model）")
      }
    }
    list.sort();
    list.dedup();
    for m in list {
      let model_id = m.trim();
      if model_id.is_empty() {
        continue;
      }
      let byok_id = format!("byok:{}:{model_id}", p.id().trim());
      let display_name = format!("{}: {model_id}", p.id().trim());
      registry.insert(
        display_name.clone(),
        serde_json::Value::String(byok_id.clone()),
      );
      info_registry.insert(
        byok_id.clone(),
        serde_json::json!({ "description": "", "disabled": false, "displayName": display_name, "shortName": display_name }),
      );
      models.push(serde_json::json!({ "name": byok_id, "suggested_prefix_char_count": 0, "suggested_suffix_char_count": 0 }));
    }
  }

  if registry.is_empty() {
    return (
      StatusCode::BAD_REQUEST,
      axum::Json(
        serde_json::json!({ "ok": false, "error": "BYOK models 为空：请检查 byok.providers[].default_model 或网络/鉴权" }),
      ),
    )
      .into_response();
  }

  let default_chat_model_id = normalize_string(active.default_model());
  if default_chat_model_id.is_empty() {
    return (
      StatusCode::BAD_REQUEST,
      axum::Json(
        serde_json::json!({ "ok": false, "error": format!("Provider({}) 缺少 default_model", active.id()) }),
      ),
    )
      .into_response();
  }
  let agent_chat_model = format!("byok:{}:{}", active.id().trim(), default_chat_model_id);

  let registry_json = serde_json::to_string(&registry).unwrap_or_else(|_| "{}".to_string());
  let info_registry_json =
    serde_json::to_string(&info_registry).unwrap_or_else(|_| "{}".to_string());

  let flags_value = obj
    .entry("feature_flags")
    .or_insert_with(|| serde_json::json!({}));
  if !flags_value.is_object() {
    *flags_value = serde_json::json!({});
  }
  let flags = flags_value
    .as_object_mut()
    .expect("feature_flags is object");
  flags.insert(
    "additional_chat_models".to_string(),
    serde_json::Value::String(registry_json.clone()),
  );
  flags.insert(
    "additionalChatModels".to_string(),
    serde_json::Value::String(registry_json.clone()),
  );
  flags.insert(
    "agent_chat_model".to_string(),
    serde_json::Value::String(agent_chat_model.clone()),
  );
  flags.insert(
    "agentChatModel".to_string(),
    serde_json::Value::String(agent_chat_model.clone()),
  );
  flags.insert(
    "enable_model_registry".to_string(),
    serde_json::Value::Bool(true),
  );
  flags.insert(
    "enableModelRegistry".to_string(),
    serde_json::Value::Bool(true),
  );
  flags.insert(
    "model_registry".to_string(),
    serde_json::Value::String(registry_json.clone()),
  );
  flags.insert(
    "modelRegistry".to_string(),
    serde_json::Value::String(registry_json.clone()),
  );
  flags.insert(
    "model_info_registry".to_string(),
    serde_json::Value::String(info_registry_json.clone()),
  );
  flags.insert(
    "modelInfoRegistry".to_string(),
    serde_json::Value::String(info_registry_json),
  );
  flags.insert(
    "show_thinking_summary".to_string(),
    serde_json::Value::Bool(true),
  );
  flags.insert(
    "showThinkingSummary".to_string(),
    serde_json::Value::Bool(true),
  );
  flags.insert(
    "fraud_sign_endpoints".to_string(),
    serde_json::Value::Bool(false),
  );
  flags.insert(
    "fraudSignEndpoints".to_string(),
    serde_json::Value::Bool(false),
  );

  obj.insert(
    "default_model".to_string(),
    serde_json::Value::String(agent_chat_model),
  );
  obj.insert("models".to_string(), serde_json::Value::Array(models));
  (StatusCode::OK, axum::Json(upstream)).into_response()
}

async fn forward_to_official(
  state: &AppState,
  cfg: &Config,
  method: axum::http::Method,
  uri: &axum::http::Uri,
  headers: &HeaderMap,
  body_bytes: Bytes,
  timeout: Duration,
) -> Response<Body> {
  let url = match build_official_url(&cfg.official.base_url, uri) {
    Ok(u) => u,
    Err(err) => {
      let mut resp = Response::new(Body::from(format!("Bad gateway: {err}")));
      *resp.status_mut() = StatusCode::BAD_GATEWAY;
      return resp;
    }
  };

  let official_token = normalize_raw_token(&cfg.official.api_token);
  if official_token.is_empty() {
    let mut resp = Response::new(Body::from("Bad gateway: official.api_token empty"));
    *resp.status_mut() = StatusCode::BAD_GATEWAY;
    return resp;
  }

  let mut out_req = state
    .http
    .request(method, url)
    .timeout(timeout)
    .header("authorization", format!("Bearer {official_token}"));

  for (k, v) in headers.iter() {
    if should_drop_forward_header(k.as_str()) || k.as_str().eq_ignore_ascii_case("authorization") {
      continue;
    }
    out_req = out_req.header(k, v);
  }

  let upstream = match out_req.body(body_bytes).send().await {
    Ok(v) => v,
    Err(err) => {
      let mut resp = Response::new(Body::from(format!("Bad gateway: {err}")));
      *resp.status_mut() = StatusCode::BAD_GATEWAY;
      return resp;
    }
  };

  let status = upstream.status();
  let headers = upstream.headers().clone();
  let stream = upstream
    .bytes_stream()
    .map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));

  let mut resp = Response::new(Body::from_stream(stream));
  *resp.status_mut() = status;
  for (k, v) in headers.iter() {
    if should_drop_response_header(k.as_str()) {
      continue;
    }
    resp.headers_mut().insert(k.clone(), v.clone());
  }
  resp
}

async fn proxy_fallback(State(state): State<AppState>, req: Request<Body>) -> Response<Body> {
  let (parts, body) = req.into_parts();
  let cfg = state.cfg.read().await.clone();
  let mode = read_byok_mode(&parts.headers);

  if !is_authorized(&parts.headers, &cfg.proxy.auth_token) {
    let present = auth_present_headers(&parts.headers);
    warn!(present=?present, "fallback 未授权（缺少或错误的鉴权 token）");
    let mut resp = Response::new(Body::from("Unauthorized"));
    *resp.status_mut() = StatusCode::UNAUTHORIZED;
    return resp;
  }

  if mode == ByokMode::Disabled {
    let mut resp = Response::new(Body::from("Disabled by routing rule"));
    *resp.status_mut() = StatusCode::NOT_FOUND;
    return resp;
  }
  if mode == ByokMode::Byok {
    let mut resp = Response::new(Body::from("BYOK endpoint not implemented"));
    *resp.status_mut() = StatusCode::NOT_IMPLEMENTED;
    return resp;
  }

  let body_bytes = match to_bytes(body, 16 * 1024 * 1024).await {
    Ok(b) => b,
    Err(err) => {
      let mut resp = Response::new(Body::from(format!("Bad request body: {err}")));
      *resp.status_mut() = StatusCode::BAD_REQUEST;
      return resp;
    }
  };
  forward_to_official(
    &state,
    &cfg,
    parts.method,
    &parts.uri,
    &parts.headers,
    body_bytes,
    Duration::from_secs(120),
  )
  .await
}

fn convert_event_to_chunks(
  state: &mut AnthropicStreamState,
  event: AnthropicStreamEvent,
) -> Vec<AugmentStreamChunk> {
  if let Some(u) = event
    .message
    .as_ref()
    .map(|m| &m.usage)
    .or(event.usage.as_ref())
  {
    state.on_usage(u);
  }

  match event.event_type.as_str() {
    "content_block_start" => {
      let Some(block) = event.content_block else {
        return Vec::new();
      };
      match block.block_type.as_str() {
        "tool_use" => {
          let (Some(id), Some(name)) = (block.id, block.name) else {
            return Vec::new();
          };
          state.on_tool_use_block_start(&id, &name);
        }
        "thinking" => state.on_thinking_block_start(),
        _ => {}
      }
      Vec::new()
    }
    "content_block_delta" => {
      let Some(delta) = event.delta else {
        return Vec::new();
      };
      match delta.delta_type.as_str() {
        "text_delta" => delta
          .text
          .as_deref()
          .map(|t| vec![state.on_text_delta(t)])
          .unwrap_or_default(),
        "input_json_delta" => {
          if let Some(partial) = delta.partial_json.as_deref() {
            state.on_tool_input_json_delta(partial);
          }
          Vec::new()
        }
        "thinking_delta" => {
          if let Some(thinking) = delta.thinking.as_deref() {
            state.on_thinking_delta(thinking);
          }
          Vec::new()
        }
        _ => Vec::new(),
      }
    }
    "content_block_stop" => {
      let mut out: Vec<AugmentStreamChunk> = Vec::new();
      if let Some(chunk) = state.on_thinking_block_stop() {
        out.push(chunk);
      }
      out.extend(state.on_tool_use_block_stop());
      out
    }
    "message_delta" => {
      let Some(delta) = event.delta else {
        return Vec::new();
      };
      let Some(reason) = delta.stop_reason.as_deref() else {
        return Vec::new();
      };
      state.on_stop_reason(reason);
      Vec::new()
    }
    "message_stop" => Vec::new(),
    "error" => vec![error_response("❌ 上游返回 error event")],
    _ => Vec::new(),
  }
}

fn ndjson_response(chunk: AugmentStreamChunk) -> Response<Body> {
  let line = serde_json::to_string(&chunk)
    .unwrap_or_else(|_| "{\"text\":\"\",\"stop_reason\":1}".to_string());
  let mut response = Response::new(Body::from(format!("{line}\n")));
  let headers = response.headers_mut();
  headers.insert(
    "content-type",
    HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
  );
  response
}

fn truncate_for_log(mut s: String, max_bytes: usize) -> String {
  if s.len() <= max_bytes {
    return s;
  }
  s.truncate(max_bytes);
  s.push_str("…<truncated>");
  s
}

fn redact_json_for_log(v: &mut serde_json::Value) {
  match v {
    serde_json::Value::Object(map) => {
      for (k, v) in map.iter_mut() {
        let key = k.trim().to_ascii_lowercase();
        if key == "prefix"
          || key == "suffix"
          || key == "selected_code"
          || key == "blobs"
          || key == "chat_history"
          || key == "chathistory"
          || key == "nodes"
          || key == "request_nodes"
          || key == "requestnodes"
          || key == "response_nodes"
          || key == "responsenodes"
          || key == "structured_request_nodes"
          || key == "structuredrequestnodes"
          || key == "structured_output_nodes"
          || key == "structuredoutputnodes"
          || key == "rules"
          || key == "tool_definitions"
        {
          let meta = match v {
            serde_json::Value::String(s) => format!("[omitted {key} len={}]", s.len()),
            serde_json::Value::Array(a) => format!("[omitted {key} len={}]", a.len()),
            serde_json::Value::Object(m) => format!("[omitted {key} keys={}]", m.len()),
            _ => format!("[omitted {key}]"),
          };
          *v = serde_json::Value::String(meta);
          continue;
        }
        if key == "authorization" || key.ends_with("api_key") || key.contains("api_key") {
          *v = serde_json::Value::String("[redacted]".to_string());
          continue;
        }
        if key == "encrypted_data" {
          let len = v.as_str().map(|s| s.len()).unwrap_or(0);
          *v = serde_json::Value::String(format!("[redacted encrypted_data len={len}]"));
          continue;
        }
        if key == "iv" {
          let len = v.as_str().map(|s| s.len()).unwrap_or(0);
          *v = serde_json::Value::String(format!("[redacted iv len={len}]"));
          continue;
        }
        redact_json_for_log(v);
      }
    }
    serde_json::Value::Array(arr) => {
      for v in arr.iter_mut() {
        redact_json_for_log(v);
      }
    }
    _ => {}
  }
}

fn format_chat_stream_body_for_log(body: &[u8]) -> String {
  match serde_json::from_slice::<serde_json::Value>(body) {
    Ok(mut v) => {
      redact_json_for_log(&mut v);
      serde_json::to_string(&v).unwrap_or_else(|_| "<json stringify failed>".to_string())
    }
    Err(_) => String::from_utf8_lossy(body).to_string(),
  }
}

fn normalize_string(s: &str) -> String {
  s.trim().to_string()
}

const AUTH_HEADER_CANDIDATES: [&str; 10] = [
  "authorization",
  "x-api-key",
  "x-api-token",
  "x-augment-api-token",
  "x-augment-token",
  "x-augment-auth-token",
  "x-auth-token",
  "x-access-token",
  "x-authorization",
  "api-key",
];

fn auth_present_headers(headers: &HeaderMap) -> Vec<&'static str> {
  AUTH_HEADER_CANDIDATES
    .iter()
    .copied()
    .filter(|name| headers.get(*name).is_some())
    .collect()
}

fn is_authorized(headers: &HeaderMap, expected_token: &str) -> bool {
  let expected = normalize_raw_token(expected_token);
  if expected.is_empty() {
    return false;
  }
  for name in AUTH_HEADER_CANDIDATES {
    let Some(raw) = headers.get(name).and_then(|v| v.to_str().ok()) else {
      continue;
    };
    for part in raw.split(',') {
      let got = normalize_raw_token(part);
      if !got.is_empty() && got == expected {
        return true;
      }
    }
  }
  false
}

fn read_byok_mode(headers: &HeaderMap) -> ByokMode {
  let raw = headers
    .get("x-byok-mode")
    .or_else(|| headers.get("x-augment-byok-mode"))
    .and_then(|v| v.to_str().ok())
    .unwrap_or("")
    .trim()
    .to_ascii_lowercase();

  match raw.as_str() {
    "byok" => ByokMode::Byok,
    "official" | "upstream" => ByokMode::Official,
    "disabled" | "off" => ByokMode::Disabled,
    _ => ByokMode::Default,
  }
}

fn read_byok_model_override(headers: &HeaderMap) -> Option<String> {
  headers
    .get("x-byok-model")
    .and_then(|v| v.to_str().ok())
    .map(str::trim)
    .filter(|s| !s.is_empty())
    .map(str::to_string)
}

fn parse_byok_model_id(model: &str) -> Option<(String, String)> {
  let s = model.trim();
  let rest = s.strip_prefix("byok:")?;
  let mut it = rest.splitn(2, ':');
  let provider_id = it.next()?.trim();
  let model_id = it.next()?.trim();
  if provider_id.is_empty() || model_id.is_empty() {
    return None;
  }
  Some((provider_id.to_string(), model_id.to_string()))
}

fn pick_active_provider(cfg: &Config) -> anyhow::Result<ProviderRef<'_>> {
  if cfg.byok.providers.is_empty() {
    anyhow::bail!("byok.providers 为空");
  }
  if let Some(id) = cfg
    .byok
    .active_provider_id
    .as_deref()
    .map(str::trim)
    .filter(|s| !s.is_empty())
  {
    return get_provider_by_id(cfg, id);
  }
  match cfg.byok.providers.first().expect("providers not empty") {
    ProviderConfig::Anthropic(p) => Ok(ProviderRef::Anthropic(p)),
    ProviderConfig::OpenAICompatible(p) => Ok(ProviderRef::OpenAICompatible(p)),
  }
}

fn get_provider_by_id<'a>(cfg: &'a Config, provider_id: &str) -> anyhow::Result<ProviderRef<'a>> {
  let pid = provider_id.trim();
  if pid.is_empty() {
    anyhow::bail!("provider_id 为空");
  }
  for p in &cfg.byok.providers {
    match p {
      ProviderConfig::Anthropic(p) if p.id.trim() == pid => return Ok(ProviderRef::Anthropic(p)),
      ProviderConfig::OpenAICompatible(p) if p.id.trim() == pid => {
        return Ok(ProviderRef::OpenAICompatible(p))
      }
      _ => {}
    }
  }
  Err(anyhow::anyhow!("未找到 provider: {pid}"))
}

fn should_drop_forward_header(name: &str) -> bool {
  let n = name.to_ascii_lowercase();
  if n == "authorization" {
    return true;
  }
  if n.starts_with("x-byok-") {
    return true;
  }
  if n == "host"
    || n == "connection"
    || n == "content-length"
    || n == "transfer-encoding"
    || n == "upgrade"
    || n == "proxy-authenticate"
    || n == "proxy-authorization"
    || n == "te"
    || n == "trailer"
  {
    return true;
  }
  if n.starts_with("x-signature-") {
    return true;
  }
  false
}

fn should_drop_response_header(name: &str) -> bool {
  let n = name.to_ascii_lowercase();
  n == "connection"
    || n == "transfer-encoding"
    || n == "keep-alive"
    || n == "proxy-authenticate"
    || n == "proxy-authorization"
    || n == "te"
    || n == "trailer"
    || n == "upgrade"
}

fn build_official_url(base_url: &str, uri: &axum::http::Uri) -> anyhow::Result<String> {
  let path = uri.path();
  let mut url = if path.trim() == "/" || path.trim().is_empty() {
    base_url.trim().to_string()
  } else {
    join_url(base_url, path)?
  };
  if let Some(q) = uri.query() {
    url.push('?');
    url.push_str(q);
  }
  let _ = url::Url::parse(&url)?;
  Ok(url)
}

fn provider_kind(provider: ProviderRef<'_>) -> &'static str {
  match provider {
    ProviderRef::Anthropic(_) => "anthropic",
    ProviderRef::OpenAICompatible(_) => "openai_compatible",
  }
}

async fn get_provider_models_cached(
  state: &AppState,
  provider: ProviderRef<'_>,
) -> anyhow::Result<Vec<String>> {
  const TTL_MS: u64 = 10 * 60_000;
  let pid = provider.id().trim().to_string();
  let kind = provider_kind(provider).to_string();
  let base_url = provider.base_url().trim().to_string();
  let now = now_ms();

  {
    let cache = state.models_cache.read().await;
    if let Some(entry) = cache.providers.get(&pid) {
      if entry.kind == kind
        && entry.base_url == base_url
        && now.saturating_sub(entry.updated_at_ms) <= TTL_MS
      {
        return Ok(entry.models.clone());
      }
    }
  }

  let models = fetch_provider_models(state, provider).await?;
  {
    let mut cache = state.models_cache.write().await;
    cache.providers.insert(
      pid,
      ModelCacheEntry {
        kind,
        base_url,
        updated_at_ms: now,
        models: models.clone(),
      },
    );
  }
  Ok(models)
}

async fn fetch_provider_models(
  state: &AppState,
  provider: ProviderRef<'_>,
) -> anyhow::Result<Vec<String>> {
  match provider {
    ProviderRef::Anthropic(p) => fetch_anthropic_models(state, p).await,
    ProviderRef::OpenAICompatible(p) => fetch_openai_models(state, p).await,
  }
}

async fn fetch_anthropic_models(
  state: &AppState,
  provider: &AnthropicProviderConfig,
) -> anyhow::Result<Vec<String>> {
  let url = join_url(&provider.base_url, "models").context("构建 Anthropic models URL 失败")?;
  let key = normalize_raw_token(&provider.api_key);
  if key.is_empty() {
    anyhow::bail!("Provider({}) api_key 为空", provider.id);
  }

  let mut req = state
    .http
    .get(url)
    .header("x-api-key", key.clone())
    .header("anthropic-version", "2023-06-01")
    .header("authorization", format!("Bearer {key}"))
    .timeout(Duration::from_secs(12));

  for (k, v) in &provider.extra_headers {
    if let Ok(value) = HeaderValue::from_str(v) {
      req = req.header(k, value);
    }
  }

  let resp = req.send().await.context("请求 Anthropic /models 失败")?;
  let status = resp.status();
  let text = resp.text().await.unwrap_or_default();
  if !status.is_success() {
    anyhow::bail!("Anthropic /models 返回错误: {status} {text}");
  }
  let json: serde_json::Value =
    serde_json::from_str(&text).context("Anthropic /models 响应不是 JSON")?;
  let data = json
    .get("data")
    .and_then(|v| v.as_array())
    .context("Anthropic /models 缺少 data[]")?;
  let mut models: Vec<String> = data
    .iter()
    .filter_map(|m| {
      m.get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
    })
    .filter(|s| !s.is_empty())
    .collect();
  models.sort();
  models.dedup();
  Ok(models)
}

async fn fetch_openai_models(
  state: &AppState,
  provider: &OpenAICompatibleProviderConfig,
) -> anyhow::Result<Vec<String>> {
  let url = join_url(&provider.base_url, "models").context("构建 OpenAI models URL 失败")?;
  let key = normalize_raw_token(&provider.api_key);
  if key.is_empty() {
    anyhow::bail!("Provider({}) api_key 为空", provider.id);
  }

  let mut req = state
    .http
    .get(url)
    .header("authorization", format!("Bearer {key}"))
    .timeout(Duration::from_secs(12));

  for (k, v) in &provider.extra_headers {
    if let Ok(value) = HeaderValue::from_str(v) {
      req = req.header(k, value);
    }
  }

  let resp = req.send().await.context("请求 OpenAI /models 失败")?;
  let status = resp.status();
  let text = resp.text().await.unwrap_or_default();
  if !status.is_success() {
    anyhow::bail!("OpenAI /models 返回错误: {status} {text}");
  }

  let json: serde_json::Value =
    serde_json::from_str(&text).context("OpenAI /models 响应不是 JSON")?;
  let data = json
    .get("data")
    .and_then(|v| v.as_array())
    .context("OpenAI /models 缺少 data[]")?;
  let mut models: Vec<String> = data
    .iter()
    .filter_map(|m| {
      m.get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
    })
    .filter(|s| !s.is_empty())
    .collect();
  models.sort();
  models.dedup();
  Ok(models)
}

fn parse_augment_request(body: &[u8]) -> anyhow::Result<AugmentRequest> {
  let value: serde_json::Value = serde_json::from_slice(body).context("解析 JSON 失败")?;
  let Some(obj) = value.as_object() else {
    anyhow::bail!("请求体不是 JSON object");
  };
  if obj.contains_key("encrypted_data") {
    anyhow::bail!("检测到 encrypted_data：本 proxy 已不支持该加密包裹；请使用官方扩展版本（或升级/更换你的扩展）");
  }
  let mut deserializer = serde_json::Deserializer::from_slice(body);
  let augment: AugmentRequest =
    serde_path_to_error::deserialize(&mut deserializer).map_err(|err| {
      let path = err.path().to_string();
      let path = if path.trim().is_empty() {
        "<root>".to_string()
      } else {
        path
      };
      let inner = err.inner().to_string();
      anyhow::anyhow!("字段 {path}: {inner}")
    })?;
  Ok(augment)
}

#[cfg(test)]
mod parse_tests {
  use super::parse_augment_request;

  #[test]
  fn parse_plain_augment_request_ok() {
    let body = br#"{"message":"hi","chat_history":[]}"#;
    let req = parse_augment_request(body).unwrap();
    assert_eq!(req.message, "hi".to_string());
  }

  #[test]
  fn parse_plain_with_null_strings_ok() {
    let body = br#"{"message":"hi","chat_history":[],"agent_memories":null,"mode":null,"prefix":null,"suffix":null,"lang":null,"path":null,"user_guidelines":null}"#;
    let req = parse_augment_request(body).unwrap();
    assert_eq!(req.message, "hi".to_string());
    assert_eq!(req.agent_memories, "");
    assert_eq!(req.mode, "");
    assert_eq!(req.prefix, "");
    assert_eq!(req.suffix, "");
    assert_eq!(req.lang, "");
    assert_eq!(req.path, "");
    assert_eq!(req.user_guidelines, "");
  }

  #[test]
  fn parse_tool_definitions_null_fields_ok() {
    let body = br#"{"message":"hi","chat_history":[],"tool_definitions":[{"name":"read_file","description":null,"input_schema":{"type":"object"},"input_schema_json":null}]}"#;
    let req = parse_augment_request(body).unwrap();
    assert_eq!(req.tool_definitions.len(), 1);
    assert_eq!(req.tool_definitions[0].name, "read_file".to_string());
    assert_eq!(req.tool_definitions[0].description, "");
    assert_eq!(req.tool_definitions[0].input_schema_json, "");
  }

  #[test]
  fn parse_tool_definitions_null_name_ok() {
    let body = br#"{"message":"hi","chat_history":[],"tool_definitions":[{"name":null,"description":"d","input_schema":{"type":"object"}}]}"#;
    let req = parse_augment_request(body).unwrap();
    assert_eq!(req.tool_definitions.len(), 1);
    assert_eq!(req.tool_definitions[0].name, "");
  }

  #[test]
  fn parse_nodes_null_text_node_content_ok() {
    let body = br#"{"message":"hi","chat_history":[],"nodes":[{"id":1,"type":0,"text_node":{"content":null}}]}"#;
    let req = parse_augment_request(body).unwrap();
    assert_eq!(req.nodes.len(), 1);
    assert_eq!(req.nodes[0].id, 1);
    assert_eq!(req.nodes[0].node_type, 0);
    assert_eq!(req.nodes[0].text_node.as_ref().unwrap().content, "");
  }

  #[test]
  fn parse_encrypted_wrapper_without_plain_fails() {
    let body = br#"{"encrypted_data":"deadbeef"}"#;
    let err = parse_augment_request(body).unwrap_err();
    assert!(err.to_string().contains("encrypted_data"));
  }
}
