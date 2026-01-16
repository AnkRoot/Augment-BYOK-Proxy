# Augment-BYOK-Proxy（Rust）

把本代理作为 Augment 扩展的 `completionURL` 使用：
- `/chat-stream`：按所选 `byok.providers[].type` 做协议转换：Anthropic（`POST {base_url}/messages` SSE）或 OpenAI-compatible（`POST {base_url}/chat/completions` SSE）；输出 Augment 期望的 NDJSON（每行一个 `{text,nodes,stop_reason}`）。
- `/get-models`：请求官方 `/get-models`，并注入 BYOK 模型 registry（`byok:<providerId>:<modelId>`），让主面板 Model Picker 可选/可切换。
- 部分 LLM 端点支持 BYOK/Official/Disabled 路由：当扩展侧注入 `x-byok-mode: byok|official|disabled` 时，优先按该模式处理（BYOK 可用 `x-byok-model` 指定 `byok:<providerId>:<modelId>`）。
- 其它所有路径：原样反代到官方 `official.base_url`（由 Rust 统一携带 `official.api_token`）。

## 快速开始

1) 配置：`cp config.example.yaml config.yaml`（按注释填写；全部配置仅来自 `config.yaml`）  
2) 启动：Rust `cargo run --release -- --config config.yaml`；预编译（GitHub Releases 解压）`./augment-byok-proxy --config config.yaml`（macOS 无权限先 `chmod +x augment-byok-proxy`；Gatekeeper：`xattr -dr com.apple.quarantine augment-byok-proxy`；Windows：`.\augment-byok-proxy.exe --config config.yaml`）  
3) VS Code（注入版扩展）配置 `completionURL/apiToken`（`apiToken` 是本代理鉴权 token，不是 LLM key）：  

```jsonc
{
  "augment.advanced": {
    "completionURL": "http://127.0.0.1:8317/",
    "apiToken": "proxy_your_auth_token"
  }
}
```

自测：`GET http://127.0.0.1:8317/health` → `{"status":"ok","service":"augment-byok-proxy"}`

## 关键行为

- `proxy.auth_token` 是 VS Code 连接本代理的鉴权 token（对应 `augment.advanced.apiToken`）。
- `official.base_url` 视为完整 API 前缀，不补/抽/猜 `/api`/`/v1`；所有未实现端点全部透传到 `${official.base_url}<path>`。
- `official.api_token` 仅由 Rust 使用：用于请求官方 `/get-models` + 其它端点反代（不会暴露给 VS Code；支持 raw token / Bearer / KEY=VALUE）。
- `byok.providers[type=anthropic].base_url` 必须是完整 Anthropic API 前缀（例 `https://api.anthropic.com/v1`），内部严格拼接 `${base_url}/messages`（不猜 `/v1`；自动补齐 `/`）。
- `byok.providers[type=openai_compatible].base_url` 必须是完整 OpenAI Chat Completions API 前缀（例 `https://api.openai.com/v1`），内部严格拼接 `${base_url}/chat/completions`（不猜 `/v1`；自动补齐 `/`）。
- 模型选择：
  - 主面板 Model Picker 的候选模型来自本代理 `/get-models` 注入的 `byok:<providerId>:<modelId>`。
  - `/chat-stream` 会解析请求体 `model` 的 byok 格式，锁定 provider + modelId；若未指定则使用 `byok.active_provider_id/byok.providers[0]` 的 `default_model`。
- 请求兼容：支持 `chat_history` 还原上下文；支持工具调用（`tool_use/tool_result` 串联）；输入 nodes 支持 `type=0` text、`type=1` tool_result（支持 `content_nodes` 文本/图片）、`type=2` image(base64)，以及 `type=3..10`（会转为提示文本）。
- 上下文压缩（可选）：`history_summary.enabled=true` 时，代理会在 `chat_history` 接近上下文上限时自动触发（`trigger_strategy=auto|chars|ratio`），用 `history_summary.provider_id/model` 专用摘要模型做滚动摘要（`rolling_summary=true` 时增量更新；两者留空则默认用 active provider + default_model），并把旧 history 压成一段 `<supervisor>...` 总结后裁剪（client 无感；会增加一次上游调用延迟；依赖请求体 `conversationId` 做缓存复用）。如需连「client→proxy」也不再传全量 history，可在注入版 VSIX 上开启 `history_summary.client_compaction_enabled=true`。
- 请求解析：显式 `null` 的字符串字段按缺省值处理；解析失败错误会附带 JSON 字段路径（便于定位是哪一个字段触发 `null → string`）。
- 日志：`logging.filter` 控制过滤；`logging.dump_chat_stream_body=true` 输出已脱敏请求摘要（不截断；仍可能包含代码片段）；请求解析失败时会额外输出该摘要用于排查。
- 扩展隐藏配置 `augment.advanced.chat.override.*` 仅进入请求体 `third_party_override`（不会直接改变请求 URL）。

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/get-models` | 获取模型列表（上游官方 + 注入 BYOK registry） |
| POST | `/chat-stream` | 核心：chat 流（Augment NDJSON） |
| POST | `/chat` | callApi：BYOK/Official/Disabled（默认转官方） |
| POST | `/completion` | callApi：BYOK/Official/Disabled（默认转官方） |
| POST | `/chat-input-completion` | callApi：BYOK/Official/Disabled（默认转官方） |
| POST | `/edit` | callApi：BYOK/Official/Disabled（默认转官方） |
| POST | `/prompt-enhancer` | callApiStream：BYOK/Official/Disabled（默认转官方） |
| POST | `/instruction-stream` | callApiStream：BYOK/Official/Disabled（默认转官方） |
| POST | `/smart-paste-stream` | callApiStream：BYOK/Official/Disabled（默认转官方） |
| POST | `/generate-commit-message-stream` | callApiStream：BYOK/Official/Disabled（默认转官方） |
| POST | `/generate-conversation-title` | callApiStream：BYOK/Official/Disabled（默认转官方） |
| ANY | `/*` | 其它端点：原样反代到官方（携带 `official.api_token`） |
| GET | `/admin` | Web 管理台（运行时编辑配置） |
| GET | `/admin/api/config` | 读取当前运行时配置（JSON） |
| PUT | `/admin/api/config` | 热更新运行时配置（JSON；不支持改监听地址/端口/日志 filter） |
| POST | `/admin/api/config/save` | 保存当前配置到启动时的 `config.yaml` |

## 管理台（可选）

访问 `http://127.0.0.1:8317/admin`，直接编辑运行时 JSON 配置；**热更新仅影响后续请求**，且：

- 监听地址/端口、`logging.filter` 变更需要重启（管理台会拒绝该类热更新）。
- `保存到文件` 会覆盖写回启动时的 `config.yaml`（注释会丢）。
- 管理台会显示 `token/api_key`，建议仅监听 `127.0.0.1`。

## 转换规则（Anthropic SSE → Augment NDJSON）

- `text_delta` → `text` + `nodes[].type=0`（`content=delta`）
- `thinking_delta` → 缓冲 → `nodes[].type=8`（在 `content_block_stop` 一次性发出；`thinking.summary`）
- `tool_use` + `input_json_delta` → 缓冲 → `nodes[].type=7`（TOOL_USE_START）+ `nodes[].type=5`（TOOL_USE）（在 `content_block_stop` 发出；都携带 `tool_use{tool_use_id,tool_name,input_json}`）
- `usage/message.usage` → `nodes[].type=10`（TOKEN_USAGE；`token_usage.*`）
- 结束：最后一行输出 `stop_reason`，并可选附带 `nodes[].type=2`（MAIN_TEXT_FINISHED；`content=full_text`）
- `stop_reason`：`end_turn/stop_sequence→1`、`max_tokens→2`、`tool_use→3`（其它默认 `1`）

## 转换规则（OpenAI SSE → Augment NDJSON）

- `choices[].delta.content` → `text` + `nodes[].type=0`（`content=delta`）
- `choices[].delta.tool_calls[].function.arguments` → 缓冲 → `nodes[].type=7`（TOOL_USE_START）+ `nodes[].type=5`（TOOL_USE）（流结束时统一发出；携带 `tool_use{tool_use_id,tool_name,input_json}`）
- `choices[].finish_reason`：`stop→1`、`length→2`、`tool_calls/function_call→3`、`content_filter→5`（其它默认 `1`）
- `usage.prompt_tokens/completion_tokens`（如上游支持 `stream_options.include_usage`）→ `nodes[].type=10`（TOKEN_USAGE）

## VSIX Patch（可选）

目标：对官方 `augment.vscode-augment` 做最小注入，提供一个面板入口（命令）用于打开 `/admin`、刷新模型列表、配置端点路由；代理本身通过 `completionURL/apiToken` 接入，不再使用 `chatStreamForward`。

- 文件：`vsix-patch/inject-code.txt`（可选：其它注入逻辑）、`vsix-patch/byok-proxy-auth-header-inject.js`（为 completionURL 自动注入 Authorization）、`vsix-patch/byok-proxy-panel-inject.js`（注册面板命令）。
- CI：`.github/workflows/manual-build.yml`（下载官方 VSIX→注入→重打包）、`.github/workflows/build.yml`（定时构建）。
- 本地重打：依赖 `python3`（仅标准库）；`python3 scripts/repack_vsix.py`（默认输出 `dist/augment-vscode-modified-v{version}.vsix`；可用 `--in/--out/--keep-workdir`）。

## 发布（维护者）

- 方式1：打 tag（例如 `v0.1.0`）并 push；CI `.github/workflows/release-proxy.yml` 自动构建并创建 Release + 附件。
- 方式2：Actions 手动运行 `Release Augment-BYOK-Proxy`，可选传入 `tag`；留空默认使用 `Cargo.toml` 版本生成 `v{version}`。
- 方式3：GitHub UI 创建 Release（`release.created`）也会触发同一工作流上传附件。
