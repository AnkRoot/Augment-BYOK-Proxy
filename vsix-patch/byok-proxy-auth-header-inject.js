// === BYOK Proxy Auth Header Inject ===
// marker: __augment_byok_proxy_auth_header_injected
//
// 目的：当 completionURL 指向本地代理时，强制所有对该 baseURL 的请求携带 `augment.advanced.apiToken`
// 作为 `Authorization: Bearer <token>`，避免扩展侧用官方 session token/空 token 访问代理而触发 401。

(function () {
  "use strict";

  const MARKER = "__augment_byok_proxy_auth_header_injected";
  const RUNTIME_KEY = "__augment_byok_proxy_runtime_v1";
  const DEFAULT_BYOK_ENDPOINTS = new Set([
    "/get-models",
    "/chat-stream",
    "/chat",
    "/completion",
    "/chat-input-completion",
    "/edit",
    "/prompt-enhancer",
    "/instruction-stream",
    "/smart-paste-stream",
    "/generate-commit-message-stream",
    "/generate-conversation-title"
  ]);
  if (globalThis && globalThis[MARKER]) return;
  try { if (globalThis) globalThis[MARKER] = true; } catch (_) { }

  function restoreModuleExportsIfOverwritten() {
    try {
      if (typeof module !== "object" || !module) return;
      if (typeof exports !== "object" || !exports) return;
      if (module.exports && module.exports !== exports) module.exports = exports;
    } catch (_) { }
  }

  restoreModuleExportsIfOverwritten();

  function ensureRuntime() {
    try {
      if (!globalThis) return null;
      if (!globalThis[RUNTIME_KEY]) globalThis[RUNTIME_KEY] = { token: "", routing: { version: 1, rules: {} }, knownEndpoints: [], scannedEndpoints: [], models: [] };
      return globalThis[RUNTIME_KEY];
    } catch (_) {
      return null;
    }
  }

  function tryRequireVscode() {
    try { return require("vscode"); } catch (_) { return null; }
  }

  function normalizeString(v) {
    return typeof v === "string" ? v.trim() : "";
  }

  function normalizeBaseUrl(url) {
    const u = normalizeString(url);
    if (!u) return "";
    return u.endsWith("/") ? u : (u + "/");
  }

  function normalizeToken(raw) {
    let t = normalizeString(raw);
    if (!t) return "";
    const lower = t.toLowerCase();
    if (lower.startsWith("bearer ")) t = t.slice(7).trim();
    const eq = t.indexOf("=");
    if (eq > 0) {
      const k = t.slice(0, eq).trim();
      const v = t.slice(eq + 1).trim();
      const looksLikeEnv = k && v && /^[A-Z0-9_]+$/.test(k) && (/_TOKEN$|_API_TOKEN$|_KEY$|_API_KEY$/.test(k));
      if (looksLikeEnv) t = v;
    }
    return t;
  }

  function readAugmentAdvanced(vscode) {
    try {
      const cfg = vscode && vscode.workspace && vscode.workspace.getConfiguration ? vscode.workspace.getConfiguration("augment") : null;
      const adv = cfg && cfg.get ? cfg.get("advanced") : null;
      const completionURL = normalizeBaseUrl(adv && typeof adv === "object" ? adv.completionURL : "");
      const apiToken = normalizeToken(adv && typeof adv === "object" ? adv.apiToken : "");
      return { completionURL, apiToken };
    } catch (_) {
      return { completionURL: "", apiToken: "" };
    }
  }

  function normalizeEndpointPath(p) {
    const s = normalizeString(p);
    if (!s) return "";
    const out = s.startsWith("/") ? s : ("/" + s);
    return out.replace(/\/+$/, "") || "/";
  }

  function pickModeAndModel(runtime, pathname) {
    const ep = normalizeEndpointPath(pathname);
    if (!ep || ep === "/") return { mode: "", model: "" };
    const rules = runtime && runtime.routing && runtime.routing.rules && typeof runtime.routing.rules === "object" ? runtime.routing.rules : {};
    const r = rules && typeof rules === "object" ? rules[ep] : null;
    const rawMode = r && typeof r === "object" ? normalizeString(r.mode).toLowerCase() : "";
    const mode =
      rawMode === "official" || rawMode === "byok" || rawMode === "disabled"
        ? rawMode
        : (DEFAULT_BYOK_ENDPOINTS.has(ep) ? "byok" : "");
    const model = r && typeof r === "object" ? normalizeString(r.model) : "";
    return { mode, model };
  }

  function getUrlString(input) {
    try {
      if (typeof input === "string") return input;
      if (input && typeof input.href === "string") return input.href;
      if (input && typeof input.url === "string") return input.url;
      return "";
    } catch (_) {
      return "";
    }
  }

  function setAuthHeader(headers, token) {
    const t = normalizeToken(token);
    if (!t) return headers;
    const v = "Bearer " + t;

    try {
      if (headers && typeof headers.set === "function") {
        headers.set("authorization", v);
        return headers;
      }
    } catch (_) { }

    if (Array.isArray(headers)) {
      const out = [];
      for (const kv of headers) {
        if (!kv || kv.length < 2) continue;
        const k = String(kv[0] || "");
        if (k.toLowerCase() === "authorization") continue;
        out.push([kv[0], kv[1]]);
      }
      out.push(["authorization", v]);
      return out;
    }

    if (headers && typeof headers === "object") {
      const out = {};
      for (const [k, val] of Object.entries(headers)) {
        if (String(k || "").toLowerCase() === "authorization") continue;
        out[k] = val;
      }
      out.authorization = v;
      return out;
    }

    return { authorization: v };
  }

  function setHeader(headers, key, value) {
    const k = normalizeString(key);
    const v = normalizeString(value);
    if (!k) return headers;

    try {
      if (headers && typeof headers.set === "function") {
        if (v) headers.set(k, v);
        else headers.delete(k);
        return headers;
      }
    } catch (_) { }

    if (Array.isArray(headers)) {
      const out = [];
      for (const kv of headers) {
        if (!kv || kv.length < 2) continue;
        const kk = String(kv[0] || "");
        if (kk.toLowerCase() === k.toLowerCase()) continue;
        out.push([kv[0], kv[1]]);
      }
      if (v) out.push([k, v]);
      return out;
    }

    if (headers && typeof headers === "object") {
      const out = {};
      for (const [kk, vv] of Object.entries(headers)) {
        if (String(kk || "").toLowerCase() === k.toLowerCase()) continue;
        out[kk] = vv;
      }
      if (v) out[k] = v;
      return out;
    }

    if (!v) return headers;
    return { [k]: v };
  }

  function asObject(v) {
    return v && typeof v === "object" && !Array.isArray(v) ? v : null;
  }

  function asArray(v) {
    return Array.isArray(v) ? v : [];
  }

  function asString(v) {
    return typeof v === "string" ? v : "";
  }

  function asNumber(v) {
    return typeof v === "number" && Number.isFinite(v) ? v : NaN;
  }

  function safeJsonParse(s) {
    if (typeof s !== "string" || !s.trim()) return null;
    try {
      const v = JSON.parse(s);
      return asObject(v);
    } catch (_) {
      return null;
    }
  }

  function approxTokenCountFromByteLen(len) {
    const BYTES_PER_TOKEN = 4;
    const n = Math.max(0, Number(len) || 0);
    return Math.ceil(n / BYTES_PER_TOKEN);
  }

  function normalizeByokModelForMatch(raw) {
    const s = normalizeString(raw);
    if (!s) return "";
    if (s.startsWith("byok:")) {
      const rest = s.slice(5);
      const idx = rest.indexOf(":");
      if (idx > 0) return rest.slice(idx + 1).trim();
    }
    if (s.startsWith("gemini-") && s.slice(7).startsWith("claude-")) return s.slice(7);
    return s;
  }

  function resolveContextWindowTokens(hs, requestedModel) {
    const model = normalizeByokModelForMatch(requestedModel);
    if (!model) return 0;
    const overrides = asObject(hs && hs.context_window_tokens_overrides) || {};
    const keys = Object.keys(overrides).filter((k) => normalizeString(k));
    keys.sort((a, b) => b.length - a.length);
    for (const k of keys) {
      if (!model.includes(k)) continue;
      const n = Number(overrides[k]);
      if (Number.isFinite(n) && n > 0) return Math.floor(n);
    }
    const def = Number(hs && hs.context_window_tokens_default);
    return Number.isFinite(def) && def > 0 ? Math.floor(def) : 0;
  }

  function getExchangeRequestNodes(ex) {
    const e = asObject(ex) || {};
    const a = asArray(e.request_nodes || e.requestNodes);
    const b = asArray(e.structured_request_nodes || e.structuredRequestNodes);
    const c = asArray(e.nodes);
    return a.concat(b, c);
  }

  function estimateNodeSizeChars(node) {
    const n = asObject(node) || {};
    let sz = 16;
    sz += asString(n.content).length;
    const textNode = asObject(n.text_node || n.textNode);
    if (textNode) sz += asString(textNode.content).length;
    const toolResultNode = asObject(n.tool_result_node || n.toolResultNode);
    if (toolResultNode) {
      sz += asString(toolResultNode.tool_use_id || toolResultNode.toolUseId).length;
      sz += asString(toolResultNode.content).length;
      const contentNodes = asArray(toolResultNode.content_nodes || toolResultNode.contentNodes);
      for (const c of contentNodes) {
        const cc = asObject(c) || {};
        sz += 8;
        sz += asString(cc.text_content || cc.textContent).length;
        const img = asObject(cc.image_content || cc.imageContent);
        if (img) sz += asString(img.image_data || img.imageData).length;
      }
    }
    const imageNode = asObject(n.image_node || n.imageNode);
    if (imageNode) sz += asString(imageNode.image_data || imageNode.imageData).length;
    const extra = [
      n.image_id_node || n.imageIdNode,
      n.ide_state_node || n.ideStateNode,
      n.edit_events_node || n.editEventsNode,
      n.checkpoint_ref_node || n.checkpointRefNode,
      n.change_personality_node || n.changePersonalityNode,
      n.file_node || n.fileNode,
      n.file_id_node || n.fileIdNode,
      n.history_summary_node || n.historySummaryNode
    ];
    for (const v of extra) {
      if (v == null) continue;
      try { sz += JSON.stringify(v).length; } catch (_) { }
    }
    const toolUse = asObject(n.tool_use || n.toolUse);
    if (toolUse) {
      sz += asString(toolUse.tool_use_id || toolUse.toolUseId).length;
      sz += asString(toolUse.tool_name || toolUse.toolName).length;
      sz += asString(toolUse.input_json || toolUse.inputJson).length;
      sz += asString(toolUse.mcp_server_name || toolUse.mcpServerName).length;
      sz += asString(toolUse.mcp_tool_name || toolUse.mcpToolName).length;
    }
    const thinking = asObject(n.thinking);
    if (thinking) sz += asString(thinking.summary).length;
    return sz;
  }

  function estimateExchangeSizeChars(ex) {
    const e = asObject(ex) || {};
    let sz = 0;
    const reqNodes = getExchangeRequestNodes(e);
    if (reqNodes.length) {
      for (const n of reqNodes) sz += estimateNodeSizeChars(n);
    } else {
      sz += asString(e.request_message || e.requestMessage).length;
    }
    const respNodes = asArray(e.response_nodes || e.responseNodes).concat(asArray(e.structured_output_nodes || e.structuredOutputNodes));
    if (respNodes.length) {
      for (const n of respNodes) sz += estimateNodeSizeChars(n);
    } else {
      sz += asString(e.response_text || e.responseText).length;
    }
    return sz;
  }

  function estimateHistorySizeChars(history) {
    let sz = 0;
    for (const ex of asArray(history)) sz += estimateExchangeSizeChars(ex);
    return sz;
  }

  function historyContainsSummary(history) {
    for (const ex of asArray(history)) {
      for (const n of getExchangeRequestNodes(ex)) {
        const nn = asObject(n) || {};
        const t = asNumber(nn.type);
        if (t === 10 && (nn.history_summary_node || nn.historySummaryNode)) return true;
      }
    }
    return false;
  }

  function exchangeHasToolResults(ex) {
    for (const n of getExchangeRequestNodes(ex)) {
      const nn = asObject(n) || {};
      const t = asNumber(nn.type);
      if (t === 1 && (nn.tool_result_node || nn.toolResultNode)) return true;
    }
    return false;
  }

  function splitHistoryForSummary(history, tailBudgetChars, triggerChars, minTailExchanges) {
    const h = asArray(history);
    if (!h.length) return { head: [], tail: [] };
    let seen = 0;
    let headRev = [];
    let tailRev = [];
    let headChars = 0;
    let tailChars = 0;
    for (let i = h.length - 1; i >= 0; i--) {
      const ex = h[i];
      const exSz = estimateExchangeSizeChars(ex);
      if ((seen + exSz) < tailBudgetChars || tailRev.length < minTailExchanges) {
        tailRev.push(ex);
        tailChars += exSz;
      } else {
        headRev.push(ex);
        headChars += exSz;
      }
      seen += exSz;
    }
    const total = headChars + tailChars;
    if (total < triggerChars) return { head: [], tail: h.slice() };
    headRev.reverse();
    tailRev.reverse();
    return { head: headRev, tail: tailRev };
  }

  function adjustTailStartToAvoidToolResultOrphans(history, tailStart) {
    let i = Math.max(0, Number(tailStart) || 0);
    while (i < history.length) {
      if (!exchangeHasToolResults(history[i])) break;
      if (i === 0) break;
      i -= 1;
    }
    return i;
  }

  function truncateInlineText(s, maxChars) {
    const text = asString(s).trim();
    const max = Math.max(0, Number(maxChars) || 0);
    if (!max || text.length <= max) return text;
    if (max <= 3) return "...".slice(0, max);
    return text.slice(0, max - 3) + "...";
  }

  function buildAbridgedHistoryText(head, maxChars) {
    const limit = Math.max(0, Number(maxChars) || 0);
    if (!limit) return "";
    const h = asArray(head);
    const pieces = [];
    for (let i = Math.max(0, h.length - 6); i < h.length; i++) {
      const ex = asObject(h[i]) || {};
      const user = truncateInlineText(ex.request_message || ex.requestMessage, 1000);
      const assistant = truncateInlineText(ex.response_text || ex.responseText, 2000);
      if (user) pieces.push(`<user>\n${user}\n</user>`);
      if (assistant) pieces.push(`<assistant>\n${assistant}\n</assistant>`);
      if (pieces.join("\n").length > limit) break;
    }
    return truncateInlineText(pieces.join("\n"), limit);
  }

  async function readProxyConfigCached(originalFetch, completionURL, token, runtime) {
    const rt = runtime || ensureRuntime() || {};
    const cache = (rt.proxyConfigCache && typeof rt.proxyConfigCache === "object") ? rt.proxyConfigCache : { fetchedAtMs: 0, ttlMs: 10000, value: null };
    rt.proxyConfigCache = cache;

    const now = Date.now();
    const ttl = Number(cache.ttlMs);
    if (cache.value && Number.isFinite(cache.fetchedAtMs) && ttl > 0 && (now - cache.fetchedAtMs) < ttl) return cache.value;

    const base = normalizeBaseUrl(completionURL);
    if (!base) return null;
    const t = normalizeToken(token);
    if (!t) return null;

    const url = base + "admin/api/config";
    let resp;
    try {
      resp = await originalFetch(url, { method: "GET", headers: { authorization: "Bearer " + t } });
    } catch (_) {
      return null;
    }
    if (!resp || !resp.ok) return null;
    let cfg;
    try {
      cfg = await resp.json();
    } catch (_) {
      return null;
    }
    cache.value = cfg;
    cache.fetchedAtMs = now;
    return cfg;
  }

  async function runSummaryModelOnce(originalFetch, completionURL, token, summaryByokModel, prompt, chatHistory, timeoutMs) {
    const base = normalizeBaseUrl(completionURL);
    const t = normalizeToken(token);
    const model = normalizeString(summaryByokModel);
    const p = normalizeString(prompt);
    if (!base || !t || !model || !p) return "";

    const url = base + "chat-stream";
    const payload = {
      message: p,
      chat_history: asArray(chatHistory),
      conversation_id: null,
      agent_memories: "",
      mode: "",
      prefix: "",
      suffix: "",
      lang: "",
      path: "",
      user_guidelines: "",
      tool_definitions: [],
      nodes: [],
      structured_request_nodes: [],
      request_nodes: []
    };

    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    const to = Number(timeoutMs);
    const timer = controller && Number.isFinite(to) && to > 0 ? setTimeout(() => { try { controller.abort(); } catch (_) { } }, to) : null;
    try {
      const resp = await originalFetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/x-ndjson",
          authorization: "Bearer " + t,
          "x-byok-mode": "byok",
          "x-byok-model": model,
          "x-byok-proxy-internal": "history-summary"
        },
        body: JSON.stringify(payload),
        signal: controller ? controller.signal : undefined
      });
      const text = await resp.text();
      if (!resp.ok) return "";
      let out = "";
      for (const line of text.split("\n")) {
        const s = line.trim();
        if (!s) continue;
        const j = safeJsonParse(s);
        if (!j) continue;
        const delta = asString(j.text);
        if (delta) out += delta;
      }
      return out.trim();
    } catch (_) {
      return "";
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  function getConversationIdFromBody(bodyObj) {
    const b = asObject(bodyObj) || {};
    return normalizeString(b.conversation_id || b.conversationId);
  }

  function getChatHistoryRef(bodyObj) {
    const b = asObject(bodyObj) || {};
    if (Array.isArray(b.chat_history)) return { key: "chat_history", value: b.chat_history };
    if (Array.isArray(b.chatHistory)) return { key: "chatHistory", value: b.chatHistory };
    return { key: "chat_history", value: [] };
  }

  function getRequestId(ex) {
    const e = asObject(ex) || {};
    return normalizeString(e.request_id || e.requestId);
  }

  function buildHistorySummaryExchange(template, summaryText, abridgedText) {
    return {
      request_id: "vsix_history_summary",
      request_message: "",
      response_text: "",
      request_nodes: [
        {
          id: -10,
          type: 10,
          content: "",
          history_summary_node: {
            summary_text: summaryText,
            summarization_request_id: "",
            history_beginning_dropped_num_exchanges: 0,
            history_middle_abridged_text: abridgedText,
            history_end: [],
            message_template: template
          }
        }
      ],
      structured_request_nodes: [],
      nodes: [],
      response_nodes: [],
      structured_output_nodes: []
    };
  }

  async function maybeClientCompactChatStream(originalFetch, runtime, completionURL, token, pathname, init) {
    if (!init || typeof init !== "object") return null;
    const method = normalizeString(init.method || "GET").toUpperCase();
    if (method !== "POST") return null;
    if (normalizeString(pathname) !== "/chat-stream") return null;
    const bodyStr = typeof init.body === "string" ? init.body : "";
    if (!bodyStr) return null;
    if (bodyStr.includes("\"encrypted_data\"")) return null;

    const cfg = await readProxyConfigCached(originalFetch, completionURL, token, runtime);
    const hs = cfg && asObject(cfg.history_summary);
    if (!hs || !hs.enabled || !hs.client_compaction_enabled) return null;

    const bodyObj = safeJsonParse(bodyStr);
    if (!bodyObj) return null;

    const convId = getConversationIdFromBody(bodyObj);
    if (!convId) return null;

    const historyRef = getChatHistoryRef(bodyObj);
    const chatHistory = asArray(historyRef.value);
    if (!chatHistory.length) return null;
    if (historyContainsSummary(chatHistory)) return null;

    const totalChars = estimateHistorySizeChars(chatHistory) + normalizeString(bodyObj.message).length;
    const triggerStrategy = normalizeString(hs.trigger_strategy || "auto").toLowerCase();
    const triggerOnChars = Number(hs.trigger_on_history_size_chars) || 0;
    const minTailExchanges = Math.max(1, Number(hs.min_tail_exchanges) || 2);
    const tailCharsExcludeDefault = Math.max(0, Number(hs.history_tail_size_chars_to_exclude) || 0);

    const reqModel = normalizeByokModelForMatch(bodyObj.model || "");
    const cwTokens = resolveContextWindowTokens(hs, reqModel);
    const triggerRatio = Number(hs.trigger_on_context_ratio);
    const targetRatio = Number(hs.target_context_ratio);

    let decision = { triggered: false, thresholdChars: triggerOnChars, tailBudgetChars: tailCharsExcludeDefault };
    if (triggerStrategy === "chars") {
      decision.triggered = triggerOnChars > 0 && totalChars >= triggerOnChars;
    } else if (triggerStrategy === "ratio" || (triggerStrategy === "auto" && cwTokens > 0)) {
      if (cwTokens <= 0 || !Number.isFinite(triggerRatio) || triggerRatio <= 0 || triggerRatio > 1) {
        decision.triggered = triggerOnChars > 0 && totalChars >= triggerOnChars;
      } else {
        const approxTokens = approxTokenCountFromByteLen(totalChars);
        const ratio = cwTokens > 0 ? (approxTokens / cwTokens) : 1;
        if (ratio >= triggerRatio) {
          const thresholdTokens = Math.ceil(cwTokens * triggerRatio);
          const thresholdChars = Math.ceil(thresholdTokens * 4);
          const targetTokens = Number.isFinite(targetRatio) && targetRatio > 0 && targetRatio <= 1 ? Math.floor(cwTokens * targetRatio) : Math.floor(cwTokens * 0.55);
          const targetCharsBudget = Math.max(0, Math.floor(targetTokens * 4));
          const abridgedLimit = Number(hs.abridged_history_params && hs.abridged_history_params.total_chars_limit) || 10000;
          const maxSummaryTokens = Number(hs.max_tokens) || 1024;
          const summaryOverhead = abridgedLimit + (maxSummaryTokens * 4) + 4096;
          const tailBudgetChars = Math.max(0, targetCharsBudget - summaryOverhead);
          decision = { triggered: true, thresholdChars, tailBudgetChars };
        }
      }
    } else {
      decision.triggered = triggerOnChars > 0 && totalChars >= triggerOnChars;
    }

    if (!decision.triggered) return null;

    const split = splitHistoryForSummary(chatHistory, decision.tailBudgetChars, decision.thresholdChars, minTailExchanges);
    if (!split.head.length || !split.tail.length) return null;

    let tailStart = split.head.length;
    tailStart = adjustTailStartToAvoidToolResultOrphans(chatHistory, tailStart);
    if (tailStart <= 0 || tailStart >= chatHistory.length) return null;

    const boundaryRequestId = getRequestId(chatHistory[tailStart]);
    if (!boundaryRequestId) return null;

    const droppedHead = chatHistory.slice(0, tailStart);
    const tail = chatHistory.slice(tailStart);
    if (!droppedHead.length || !tail.length) return null;

    const cache = (runtime && runtime.historySummaryCache && typeof runtime.historySummaryCache === "object") ? runtime.historySummaryCache : {};
    if (runtime) runtime.historySummaryCache = cache;
    const ttlMs = Number(hs.cache_ttl_ms) || (30 * 60 * 1000);
    const now = Date.now();
    const prev = asObject(cache[convId]) || null;
    const prevOk = prev && (!ttlMs || (now - (Number(prev.updated_at_ms) || 0)) <= ttlMs);

    let summaryText = "";
    if (prevOk && normalizeString(prev.summarized_until_request_id) === boundaryRequestId) {
      summaryText = normalizeString(prev.summary_text);
    } else {
      const providerId = normalizeString(hs.provider_id);
      const modelId = normalizeString(hs.model);
      const byokModel = providerId && modelId ? ("byok:" + providerId + ":" + modelId) : "";
      const basePrompt = normalizeString(hs.prompt);
      if (!byokModel || !basePrompt) return null;

      let usedRolling = false;
      let prompt = basePrompt;
      let inputHistory = droppedHead.slice();

      if (hs.rolling_summary && prevOk && prev && normalizeString(prev.summarized_until_request_id) && normalizeString(prev.summary_text)) {
        const prevBoundary = normalizeString(prev.summarized_until_request_id);
        if (prevBoundary && prevBoundary !== boundaryRequestId) {
          const pos = chatHistory.findIndex((x) => getRequestId(x) === prevBoundary);
          if (pos >= 0 && pos < tailStart) {
            const delta = chatHistory.slice(pos, tailStart);
            if (delta.length) {
              const prevSummaryExchange = {
                request_id: "vsix_history_summary_prev",
                request_message: "[PREVIOUS_SUMMARY]\n" + normalizeString(prev.summary_text) + "\n[/PREVIOUS_SUMMARY]",
                response_text: "",
                request_nodes: [],
                structured_request_nodes: [],
                nodes: [],
                response_nodes: [],
                structured_output_nodes: []
              };
              inputHistory = [prevSummaryExchange].concat(delta);
              prompt = basePrompt + "\n\nYou will be given an existing summary and additional new conversation turns. Update the summary to include the new information. Output only the updated summary.";
              usedRolling = true;
            }
          }
        }
      }

      const maxInputChars = Number(hs.max_summarization_input_chars) || 250000;
      if (maxInputChars > 0) {
        if (usedRolling) {
          while (inputHistory.length > 1 && estimateHistorySizeChars(inputHistory) > maxInputChars) inputHistory.splice(1, 1);
        } else {
          while (inputHistory.length > 1 && estimateHistorySizeChars(inputHistory) > maxInputChars) inputHistory.shift();
        }
      }
      if (!inputHistory.length) return null;

      const timeoutSeconds = Number(hs.timeout_seconds) || 60;
      summaryText = await runSummaryModelOnce(originalFetch, completionURL, token, byokModel, prompt, inputHistory, timeoutSeconds * 1000);
      summaryText = normalizeString(summaryText);
      if (!summaryText) return null;
      cache[convId] = {
        summary_text: summaryText,
        summarized_until_request_id: boundaryRequestId,
        summarization_request_id: "",
        updated_at_ms: now
      };
    }

    const template = normalizeString(hs.summary_node_request_message_template || "");
    if (!template) return null;

    const abridged = buildAbridgedHistoryText(droppedHead, Number(hs.abridged_history_params && hs.abridged_history_params.total_chars_limit) || 10000);
    const summaryExchange = buildHistorySummaryExchange(template, summaryText, abridged);
    const newHistory = [summaryExchange].concat(tail);

    bodyObj[historyRef.key] = newHistory;
    init.body = JSON.stringify(bodyObj);
    return { ok: true, boundaryRequestId, convId };
  }

  function patchFetch(vscode) {
    const originalFetch = globalThis.fetch;
    if (typeof originalFetch !== "function") return;

    globalThis.fetch = async function (input, init) {
      try {
        const runtime = ensureRuntime();
        const { completionURL, apiToken } = readAugmentAdvanced(vscode);
        const token = normalizeToken((runtime && runtime.token) ? runtime.token : apiToken);
        const url = getUrlString(input);
        if (completionURL && typeof url === "string" && url.startsWith(completionURL)) {
          let pathname = "";
          try { pathname = normalizeEndpointPath(new URL(url).pathname); } catch (_) { }
          const picked = pickModeAndModel(runtime, pathname);
          const mode = picked.mode;
          const model = picked.model;

          try {
            if (init && typeof init === "object") {
              const headers = init.headers;
              const internal = headers && typeof headers === "object" ? (headers["x-byok-proxy-internal"] || headers["X-Byok-Proxy-Internal"]) : "";
              if (!internal) {
                await maybeClientCompactChatStream(originalFetch, runtime, completionURL, token, pathname, init);
              }
            }
          } catch (_) { }

          if (mode === "disabled") {
            const body = JSON.stringify({ ok: false, error: "Disabled by BYOK routing", endpoint: pathname || "" });
            return new Response(body, { status: 404, headers: { "content-type": "application/json; charset=utf-8" } });
          }

          const bearer = token ? ("Bearer " + token) : "";
          if (init && typeof init === "object") {
            init.headers = setAuthHeader(init.headers, token);
          } else {
            init = { headers: setAuthHeader(undefined, token) };
          }
          if (mode === "official" || mode === "byok") init.headers = setHeader(init.headers, "x-byok-mode", mode);
          init.headers = (mode === "byok" && model) ? setHeader(init.headers, "x-byok-model", model) : setHeader(init.headers, "x-byok-model", "");
          try {
            if (input && input.headers && typeof input.headers.set === "function") {
              input.headers.set("authorization", bearer);
            }
          } catch (_) { }
        }
      } catch (_) { }
      return originalFetch.call(this, input, init);
    };
  }

  const vscode = tryRequireVscode();
  patchFetch(vscode);
})();

// === BYOK Proxy Auth Header Inject End ===
