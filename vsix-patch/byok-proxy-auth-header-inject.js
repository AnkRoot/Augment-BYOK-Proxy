// === BYOK Proxy Auth Header Inject ===
// marker: __augment_byok_proxy_auth_header_injected
//
// 目的：当 completionURL 指向本地代理时，强制所有对该 baseURL 的请求携带 `augment.advanced.apiToken`
// 作为 `Authorization: Bearer <token>`，避免扩展侧用官方 session token/空 token 访问代理而触发 401。
//
// 注意：history_summary（上下文压缩/摘要）统一由 proxy(Rust) 侧负责，本注入脚本不再做 client-side compaction，
// 以避免双实现导致行为漂移、节点不一致、缓存策略分叉等维护成本。
//
// 说明：本脚本仅做三件事：
// 1) 强制补齐 Authorization
// 2) 按 runtime.routing.rules 决定 x-byok-mode / x-byok-model
// 3) mode=disabled 时直接返回 404（保持与代理侧路由一致）

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
      if (!globalThis[RUNTIME_KEY]) {
        globalThis[RUNTIME_KEY] = {
          token: "",
          routing: { version: 1, rules: {} },
          knownEndpoints: [],
          scannedEndpoints: [],
          models: []
        };
      }
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

          if (mode === "disabled") {
            const body = JSON.stringify({ ok: false, error: "Disabled by BYOK routing", endpoint: pathname || "" });
            return new Response(body, { status: 404, headers: { "content-type": "application/json; charset=utf-8" } });
          }

          if (init && typeof init === "object") {
            init.headers = setAuthHeader(init.headers, token);
          } else {
            init = { headers: setAuthHeader(undefined, token) };
          }
          if (mode === "official" || mode === "byok") init.headers = setHeader(init.headers, "x-byok-mode", mode);
          init.headers = (mode === "byok" && model) ? setHeader(init.headers, "x-byok-model", model) : setHeader(init.headers, "x-byok-model", "");

          try {
            if (input && input.headers && typeof input.headers.set === "function") {
              const bearer = token ? ("Bearer " + token) : "";
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
