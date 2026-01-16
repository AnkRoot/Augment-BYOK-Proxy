// === BYOK Proxy Panel Inject ===
// marker: __augment_byok_proxy_panel_injected

(function () {
  "use strict";

  const MARKER = "__augment_byok_proxy_panel_injected";
  const COMMAND_ID = "vscode-augment.byokProxy.settings";
  const VIEW_TYPE = "augmentByokProxyPanel";
  const TITLE = "Augment BYOK Proxy";
  const RUNTIME_KEY = "__augment_byok_proxy_runtime_v1";
  const STATE_ROUTING_KEY = "__augment_byok_proxy_v1.routing";
  const IMPLEMENTED_ENDPOINTS = [
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
  ];

  if (globalThis && globalThis[MARKER]) return;
  try { if (globalThis) globalThis[MARKER] = true; } catch (_) { }

  function tryRequireVscode() {
    try { return require("vscode"); } catch (_) { return null; }
  }

  function normalizeString(v) {
    return typeof v === "string" ? v.trim() : "";
  }

  function ensureRuntime() {
    try {
      if (!globalThis) return null;
      if (!globalThis[RUNTIME_KEY]) globalThis[RUNTIME_KEY] = { routing: { version: 1, rules: {} }, models: [] };
      return globalThis[RUNTIME_KEY];
    } catch (_) {
      return null;
    }
  }

  function readAugmentAdvancedConfig(vscode) {
    try {
      const cfg = vscode && vscode.workspace && vscode.workspace.getConfiguration ? vscode.workspace.getConfiguration("augment") : null;
      const adv = cfg && cfg.get ? cfg.get("advanced") : null;
      const completionURL = normalizeString(adv && typeof adv === "object" ? adv.completionURL : "");
      return { completionURL };
    } catch (_) {
      return { completionURL: "" };
    }
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function buildHtml(initial) {
    const nonce = Math.random().toString(16).slice(2) + Math.random().toString(16).slice(2);
    const csp = [
      "default-src 'none'",
      "img-src data:",
      "style-src 'unsafe-inline'",
      `script-src 'nonce-${nonce}'`,
      "connect-src http: https:"
    ].join("; ");

    const initJson = JSON.stringify({ implementedEndpoints: IMPLEMENTED_ENDPOINTS });

    return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta http-equiv="Content-Security-Policy" content="${escapeHtml(csp)}" />
  <title>${escapeHtml(TITLE)}</title>
  <style>
    body{font-family:var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background);margin:0;padding:0}
    .container{max-width:1040px;margin:0 auto;padding:16px}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
    .card{border:1px solid var(--vscode-editorWidget-border);background:var(--vscode-editorWidget-background);border-radius:6px;margin:12px 0;overflow:hidden}
    .cardHeader{padding:10px 12px;border-bottom:1px solid var(--vscode-editorWidget-border);display:flex;align-items:center;justify-content:space-between;gap:12px}
    .cardBody{padding:12px}
    label{display:block;font-size:12px;color:var(--vscode-descriptionForeground);margin-bottom:6px}
    input,textarea,select{width:100%;box-sizing:border-box;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border);padding:6px 8px;border-radius:4px}
    input[readonly]{opacity:0.85}
    textarea{min-height:160px;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:12px}
    button{padding:6px 10px;border:1px solid var(--vscode-button-border,transparent);background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground);cursor:pointer;border-radius:4px}
    button.primary{background:var(--vscode-button-background);color:var(--vscode-button-foreground)}
    button.danger{border-color:var(--vscode-inputValidation-errorBorder);color:var(--vscode-errorForeground)}
	    table{width:100%;border-collapse:collapse}
    th,td{border-bottom:1px solid var(--vscode-editorWidget-border);padding:8px 6px;text-align:left;vertical-align:middle}
    th{font-size:12px;color:var(--vscode-descriptionForeground);font-weight:600}
    code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:12px}
    .muted{color:var(--vscode-descriptionForeground);font-size:12px}
    pre{background:var(--vscode-textCodeBlock-background);padding:10px;white-space:pre-wrap;border-radius:4px;margin:0}
    .badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--vscode-editorWidget-border);padding:2px 8px;border-radius:999px;font-size:12px;color:var(--vscode-descriptionForeground)}
    .inlineNote{margin-top:6px}
  </style>
</head>
<body>
		  <div class="container">
		    <h2 style="margin:0 0 6px 0">${escapeHtml(TITLE)}</h2>
		    <div class="muted">管理端点路由、模型与上下文压缩；更完整的代理配置可在 <code>/admin</code> 调整。</div>

			    <div class="card">
			      <div class="cardHeader">
			        <strong>连接</strong>
			        <div class="row" style="gap:8px">
		          <span class="badge" id="modelsStatus">models: 0</span>
		          <button id="refreshModels">刷新模型</button>
		          <button id="openAdmin" class="primary">打开 /admin</button>
		          <button id="openSettings">设置</button>
		        </div>
			      </div>
		    </div>

		    <div class="card">
		      <div class="cardHeader">
		        <div>
		          <strong>上下文压缩</strong>
		          <span class="muted">（History Summary：后台滚动摘要，仅影响发送给模型）</span>
		        </div>
		        <div class="row" style="gap:8px">
		          <button id="refreshHistorySummary">刷新</button>
		          <button id="applyHistorySummary" class="primary">应用</button>
		          <button id="saveProxyConfig">保存到 config.yaml</button>
		          <button id="clearHistorySummaryCache" class="danger">清空摘要缓存</button>
		        </div>
		      </div>
		      <div class="cardBody">
		        <div class="grid">
		          <div>
		            <label>history_summary.enabled</label>
		            <div class="row" style="gap:8px">
		              <input id="hsEnabled" type="checkbox" style="width:auto" />
		              <span class="muted">启用</span>
		            </div>
		            <div class="muted inlineNote">面板/聊天 UI 仍显示完整历史；压缩仅用于发给上游模型。</div>
		          </div>
		          <div>
		            <label>history_summary.model（可选）</label>
		            <select id="hsModel"></select>
		            <div class="muted inlineNote">
		              为空则跟随当前对话模型；候选来自 <code>/get-models</code> 注入的 <code>byok:&lt;providerId&gt;:&lt;modelId&gt;</code>。
		            </div>
		          </div>
		        </div>
		        <div class="muted inlineNote">提示：缓存按 <code>conversation_id</code> 复用并持久化；删除 thread 时会自动清理对应缓存。</div>
		        <pre id="hsStatus" style="margin-top:12px">ready</pre>
		      </div>
		    </div>

		    <div class="card">
	      <div class="cardHeader">
	        <div>
	          <strong>端点路由</strong>
	          <span class="muted">（byok=走 BYOK；official=转官方；disabled=禁用；model 仅对 byok 生效）</span>
	        </div>
	        <div class="row" style="gap:8px">
	          <button id="clearRules" class="danger">清空</button>
	          <button id="saveRules" class="primary">保存</button>
	        </div>
		      </div>
		      <div class="cardBody">
	        <div class="muted" style="margin-bottom:10px">Model 下拉来自 <code>/get-models</code> 注入的 <code>byok:&lt;providerId&gt;:&lt;modelId&gt;</code>。</div>
	          <table>
	            <thead>
	              <tr>
	                <th style="width:38%">Endpoint</th>
                <th style="width:18%">Mode</th>
                <th style="width:44%">Model（byok）</th>
              </tr>
	            </thead>
	            <tbody id="rulesTbody"></tbody>
	          </table>
		      </div>
		    </div>
		  </div>

			  <script nonce="${nonce}">
			    const vscode = acquireVsCodeApi();
			    const $ = (id) => document.getElementById(id);
			    const normalize = (v) => typeof v === 'string' ? v.trim() : '';
			    const init = ${initJson};
		    const state = { rules: {}, endpoints: [], models: [], historySummary: { enabled: false, model: '' } };

		    function normalizeEndpoint(ep) {
	      const s = normalize(ep);
	      if (!s) return '';
	      const p = s.startsWith('/') ? s : ('/' + s);
	      return p.replace(/\\/+$/, '') || '/';
		    }
	
	    const implemented = new Set(Array.isArray(init.implementedEndpoints) ? init.implementedEndpoints.map(normalizeEndpoint).filter(Boolean) : []);
	
		    function defaultModeForEndpoint(ep) {
		      const e = normalizeEndpoint(ep);
		      return implemented.has(e) ? 'byok' : 'official';
		    }

	    function renderModelsStatus() {
	      const n = Array.isArray(state.models) ? state.models.length : 0;
	      $('modelsStatus').textContent = 'models: ' + String(n);
	    }

	    function groupModelsByProvider(models) {
      const out = new Map();
      for (const m of Array.isArray(models) ? models : []) {
        const raw = normalize(m);
        if (!raw.startsWith('byok:')) continue;
        const rest = raw.slice('byok:'.length);
        const idx = rest.indexOf(':');
        if (idx <= 0) continue;
        const pid = normalize(rest.slice(0, idx));
        const mid = normalize(rest.slice(idx + 1));
        if (!pid || !mid) continue;
        if (!out.has(pid)) out.set(pid, []);
        out.get(pid).push({ id: raw, modelId: mid });
      }
      for (const [k, arr] of out.entries()) arr.sort((a, b) => a.modelId.localeCompare(b.modelId));
      return Array.from(out.entries()).sort((a, b) => a[0].localeCompare(b[0]));
    }

	    function setHistorySummaryStatus(v) {
	      const el = $('hsStatus');
	      if (!el) return;
	      const s = (typeof v === 'string') ? v : JSON.stringify(v, null, 2);
	      el.textContent = s || '';
	    }

	    function renderHistorySummary() {
	      const hs = state.historySummary && typeof state.historySummary === 'object' ? state.historySummary : { enabled: false, model: '' };
	      const enabled = Boolean(hs.enabled);
	      const model = normalize(hs.model);

	      const cb = $('hsEnabled');
	      if (cb) cb.checked = enabled;

	      const sel = $('hsModel');
	      if (sel) {
	        const modelGroups = groupModelsByProvider(state.models);
	        sel.innerHTML = '';
	        sel.appendChild(new Option('(跟随当前对话模型)', ''));
	        for (const [providerId, items] of modelGroups) {
	          const og = document.createElement('optgroup');
	          og.label = providerId;
	          for (const it of items) og.appendChild(new Option(it.modelId, it.id));
	          sel.appendChild(og);
	        }
	        sel.value = model;
	      }
	    }

	    function renderRules() {
      const tbody = $('rulesTbody');
      tbody.innerHTML = '';
      const endpoints = Array.isArray(state.endpoints) ? state.endpoints.slice() : [];
      const rules = state.rules && typeof state.rules === 'object' ? state.rules : {};
      const models = Array.isArray(state.models) ? state.models : [];
      const modelSet = new Set(models.map((x) => normalize(x)).filter(Boolean));
	      const modelGroups = groupModelsByProvider(models);
	      endpoints.sort((a, b) => String(a).localeCompare(String(b)));
	      for (const ep of endpoints) {
	        const rule = rules[ep] && typeof rules[ep] === 'object' ? rules[ep] : {};
	        const defaultMode = defaultModeForEndpoint(ep);
	        const storedMode = normalize(rule.mode);
	        const mode = storedMode || defaultMode;
	        const model = normalize(rule.model) || '';

        const tr = document.createElement('tr');

        const tdEp = document.createElement('td');
        tdEp.textContent = ep;

	        const tdMode = document.createElement('td');
	        const sel = document.createElement('select');
	        sel.appendChild(new Option('byok', 'byok'));
	        sel.appendChild(new Option('official', 'official'));
	        sel.appendChild(new Option('disabled', 'disabled'));
	        sel.value = mode;
	        tdMode.appendChild(sel);

        const tdModel = document.createElement('td');
        const selModel = document.createElement('select');
        selModel.appendChild(new Option('(默认/不指定)', ''));
        if (model && !modelSet.has(model)) selModel.appendChild(new Option('(自定义) ' + model, model));
        for (const [providerId, items] of modelGroups) {
          const og = document.createElement('optgroup');
          og.label = providerId;
          for (const it of items) og.appendChild(new Option(it.modelId, it.id));
          selModel.appendChild(og);
        }
	        selModel.value = model;
	        selModel.disabled = mode !== 'byok';
	        selModel.addEventListener('change', () => {
	          const v = normalize(selModel.value);
	          if (!v) {
	            if (state.rules[ep] && typeof state.rules[ep] === 'object') {
	              delete state.rules[ep].model;
	              if (!Object.keys(state.rules[ep]).length) {
	                delete state.rules[ep];
	                if (!implemented.has(ep)) state.endpoints = state.endpoints.filter((x) => x !== ep);
	              }
	            }
	            return;
	          }
	          state.rules[ep] = { ...(state.rules[ep] || {}), model: v };
	        });
        tdModel.appendChild(selModel);

		        sel.addEventListener('change', () => {
		          const v = normalize(sel.value);
		          const nextMode = v || defaultMode;
		          if (v && v !== defaultMode) state.rules[ep] = { ...(state.rules[ep] || {}), mode: v };
		          else if (state.rules[ep] && typeof state.rules[ep] === 'object') delete state.rules[ep].mode;
		          const enabled = nextMode === 'byok';
		          selModel.disabled = !enabled;
		          if (!enabled) {
		            selModel.value = '';
		            if (state.rules[ep] && typeof state.rules[ep] === 'object') delete state.rules[ep].model;
		          }
		          if (state.rules[ep] && typeof state.rules[ep] === 'object' && !Object.keys(state.rules[ep]).length) {
		            delete state.rules[ep];
		            if (!implemented.has(ep)) state.endpoints = state.endpoints.filter((x) => x !== ep);
		          }
		          renderRules();
		        });

        tr.appendChild(tdEp);
        tr.appendChild(tdMode);
        tr.appendChild(tdModel);
        tbody.appendChild(tr);
      }
	    }

		    $('openAdmin').addEventListener('click', () => {
		      vscode.postMessage({ type: 'rpc', method: 'openAdmin', params: {} });
		    });

	    $('openSettings').addEventListener('click', () => {
	      vscode.postMessage({ type: 'rpc', method: 'openSettings', params: {} });
	    });

		    $('refreshModels').addEventListener('click', async () => {
		      vscode.postMessage({ type: 'rpc', method: 'refreshModels', params: {} });
		    });

	    $('clearRules').addEventListener('click', () => {
	      state.rules = {};
	      renderRules();
	      vscode.postMessage({ type: 'rpc', method: 'clearRouting', params: {} });
    });

		    $('saveRules').addEventListener('click', () => {
	      const rules = {};
	      for (const [k, v] of Object.entries(state.rules || {})) {
	        const ep = normalizeEndpoint(k);
	        if (!ep || ep === '/') continue;
	        const mode = normalize(v && v.mode);
	        const model = normalize(v && v.model);
	        if (!mode && !model) continue;
	        rules[ep] = {};
	        if (mode) rules[ep].mode = mode;
	        if (model) rules[ep].model = model;
	      }
	      state.rules = rules;
	      renderRules();
	      vscode.postMessage({ type: 'rpc', method: 'saveRouting', params: { rules } });
	    });

		    window.addEventListener('message', (ev) => {
		      const msg = ev && ev.data;
		      if (!msg || typeof msg !== 'object') return;
			      if (msg.type === 'state') {
			        try {
			          const d = msg.data && typeof msg.data === 'object' ? msg.data : {};
			          state.rules = (d.rules && typeof d.rules === 'object') ? d.rules : {};
			          state.endpoints = Array.isArray(d.endpoints) ? d.endpoints.map(normalizeEndpoint).filter(Boolean) : [];
			          state.models = Array.isArray(d.models) ? d.models.map(normalize).filter(Boolean) : [];
			          state.historySummary = (d.historySummary && typeof d.historySummary === 'object') ? d.historySummary : { enabled: false, model: '' };
			          if (d.historySummaryStatus != null) setHistorySummaryStatus(d.historySummaryStatus);
			          renderModelsStatus();
			          renderHistorySummary();
			          renderRules();
			        } catch (e) {
			          console.error(e);
			        }
			      }
			    });

			    $('refreshHistorySummary').addEventListener('click', () => {
			      setHistorySummaryStatus('refreshing...');
			      vscode.postMessage({ type: 'rpc', method: 'refreshHistorySummary', params: {} });
			    });

			    $('applyHistorySummary').addEventListener('click', () => {
			      const enabled = Boolean($('hsEnabled') && $('hsEnabled').checked);
			      const model = normalize($('hsModel') && $('hsModel').value);
			      setHistorySummaryStatus('applying...');
			      vscode.postMessage({ type: 'rpc', method: 'applyHistorySummary', params: { enabled, model } });
			    });

			    $('saveProxyConfig').addEventListener('click', () => {
			      setHistorySummaryStatus('saving...');
			      vscode.postMessage({ type: 'rpc', method: 'saveProxyConfig', params: {} });
			    });

			    $('clearHistorySummaryCache').addEventListener('click', () => {
			      setHistorySummaryStatus('clearing cache...');
			      vscode.postMessage({ type: 'rpc', method: 'clearHistorySummaryCache', params: {} });
			    });
			
			    vscode.postMessage({ type: 'rpc', method: 'getState', params: {} });
			  </script>
	</body>
</html>`;
  }

  function normalizeEndpoint(ep) {
    const s = normalizeString(ep);
    if (!s) return "";
    const p = s.startsWith("/") ? s : ("/" + s);
    return p.replace(/\/+$/, "") || "/";
  }

  function normalizeRouting(raw) {
    const out = { version: 1, rules: {} };
    if (!raw || typeof raw !== "object") return out;
    const rules = raw.rules && typeof raw.rules === "object" ? raw.rules : raw;
    if (!rules || typeof rules !== "object") return out;
    for (const [k, v] of Object.entries(rules)) {
      const ep = normalizeEndpoint(k);
      if (!ep || ep === "/") continue;
      const r = v && typeof v === "object" ? v : {};
      const mode = normalizeString(r.mode).toLowerCase();
      const model = normalizeString(r.model);
      const rr = {};
      if (mode === "official" || mode === "byok" || mode === "disabled") rr.mode = mode;
      if ((rr.mode === "byok" || !rr.mode) && model) rr.model = model;
      if (Object.keys(rr).length) out.rules[ep] = rr;
    }
    return out;
  }

  function joinBaseUrl(baseUrl, endpoint) {
    const b = normalizeString(baseUrl);
    const e = normalizeString(endpoint).replace(/^\/+/, "");
    if (!b || !e) return "";
    const base = b.endsWith("/") ? b : (b + "/");
    return base + e;
  }

  async function loadTokenAndRouting(context) {
    const runtime = ensureRuntime();
    if (!runtime || !context) return;
    try {
      const routingRaw = await context.globalState.get(STATE_ROUTING_KEY);
      runtime.routing = normalizeRouting(routingRaw);
    } catch (_) { }
  }

  async function saveRouting(context, nextRouting) {
    if (!context) return;
    const runtime = ensureRuntime();
    const routing = normalizeRouting(nextRouting);
    await context.globalState.update(STATE_ROUTING_KEY, routing);
    if (runtime) runtime.routing = routing;
  }

  async function clearRouting(context) {
    if (!context) return;
    const runtime = ensureRuntime();
    await context.globalState.update(STATE_ROUTING_KEY, undefined);
    if (runtime) runtime.routing = { version: 1, rules: {} };
  }

  async function fetchByokModels({ completionURL }) {
    const base = normalizeString(completionURL);
    if (!base) throw new Error("completionURL 为空");
    const url = joinBaseUrl(base, "get-models");
    const headers = { "content-type": "application/json" };
    const resp = await fetch(url, { method: "POST", headers, body: "{}" });
    const text = await resp.text().catch(() => "");
    if (!resp.ok) throw new Error(`get-models 失败: ${resp.status} ${text.slice(0, 200)}`.trim());
    const json = text ? JSON.parse(text) : null;
    const models = json && typeof json === "object" && Array.isArray(json.models) ? json.models : [];
    const names = models
      .map((m) => (m && typeof m === "object" ? normalizeString(m.name) : ""))
      .filter(Boolean)
      .filter((m) => m.startsWith("byok:"));
    names.sort();
    return names;
  }

  function parseByokModelId(raw) {
    const s = normalizeString(raw);
    if (!s.startsWith("byok:")) return null;
    const rest = s.slice("byok:".length);
    const idx = rest.indexOf(":");
    if (idx <= 0) return null;
    const providerId = normalizeString(rest.slice(0, idx));
    const modelId = normalizeString(rest.slice(idx + 1));
    if (!providerId || !modelId) return null;
    return { providerId, modelId };
  }

  async function fetchProxyConfig({ completionURL }) {
    const base = normalizeString(completionURL);
    if (!base) throw new Error("completionURL 为空");
    const url = joinBaseUrl(base, "admin/api/config");
    const resp = await fetch(url, { method: "GET" });
    const text = await resp.text().catch(() => "");
    if (!resp.ok) throw new Error(`admin/api/config 失败: ${resp.status} ${text.slice(0, 300)}`.trim());
    const json = text ? JSON.parse(text) : null;
    if (!json || typeof json !== "object") throw new Error("admin/api/config 响应不是 JSON object");
    return json;
  }

  function extractHistorySummaryState(cfg) {
    const root = cfg && typeof cfg === "object" ? cfg : null;
    const hs = root && root.history_summary && typeof root.history_summary === "object" ? root.history_summary : null;
    const enabled = Boolean(hs && hs.enabled);
    const pid = normalizeString(hs && hs.provider_id);
    const mid = normalizeString(hs && hs.model);
    const model = pid && mid ? `byok:${pid}:${mid}` : "";
    return { enabled, model };
  }

  async function applyProxyHistorySummary({ completionURL, enabled, model }) {
    const cfg = await fetchProxyConfig({ completionURL });
    if (!cfg.history_summary || typeof cfg.history_summary !== "object") cfg.history_summary = {};
    cfg.history_summary.enabled = Boolean(enabled);

    const parsed = parseByokModelId(model);
    if (!parsed) {
      cfg.history_summary.provider_id = "";
      cfg.history_summary.model = "";
    } else {
      cfg.history_summary.provider_id = parsed.providerId;
      cfg.history_summary.model = parsed.modelId;
    }

    const url = joinBaseUrl(normalizeString(completionURL), "admin/api/config");
    const resp = await fetch(url, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(cfg) });
    const text = await resp.text().catch(() => "");
    const json = text ? JSON.parse(text) : null;
    if (!resp.ok) {
      const msg = json && typeof json === "object" ? (json.error || json.message || text) : text;
      throw new Error(`应用失败: ${String(msg || resp.status).slice(0, 300)}`.trim());
    }
    return json;
  }

  async function saveProxyConfigToFile({ completionURL }) {
    const base = normalizeString(completionURL);
    if (!base) throw new Error("completionURL 为空");
    const url = joinBaseUrl(base, "admin/api/config/save");
    const resp = await fetch(url, { method: "POST" });
    const text = await resp.text().catch(() => "");
    const json = text ? JSON.parse(text) : null;
    if (!resp.ok) {
      const msg = json && typeof json === "object" ? (json.error || json.message || text) : text;
      throw new Error(`保存失败: ${String(msg || resp.status).slice(0, 300)}`.trim());
    }
    return json;
  }

  async function clearProxyHistorySummaryCacheAll({ completionURL }) {
    const base = normalizeString(completionURL);
    if (!base) throw new Error("completionURL 为空");
    const url = joinBaseUrl(base, "admin/api/history-summary-cache/clear");
    const resp = await fetch(url, { method: "POST" });
    const text = await resp.text().catch(() => "");
    const json = text ? JSON.parse(text) : null;
    if (!resp.ok) {
      const msg = json && typeof json === "object" ? (json.error || json.message || text) : text;
      throw new Error(`清空失败: ${String(msg || resp.status).slice(0, 300)}`.trim());
    }
    return json;
  }

  function installContextHook() {
    const runtime = ensureRuntime();
    if (!runtime) return;
    if (runtime.__ctxHookInstalled) return;
    runtime.__ctxHookInstalled = true;

    const moduleRef = typeof module === "object" && module ? module : null;
    if (!moduleRef) return;

    const wrapExports = (exp) => {
      try {
        if (!exp) return exp;
        if (typeof exp !== "object" && typeof exp !== "function") return exp;
        if (exp.__byokProxyExportsWrapped) return exp;
        const cache = runtime.__activateWrapperCache instanceof WeakMap ? runtime.__activateWrapperCache : (runtime.__activateWrapperCache = new WeakMap());
        return new Proxy(exp, {
          get(target, prop, receiver) {
            if (prop === "__byokProxyExportsWrapped") return true;
            if (prop === "activate") {
              const orig = Reflect.get(target, prop, receiver);
              if (typeof orig !== "function") return orig;
              const cached = cache.get(orig);
              if (cached) return cached;
              const wrapped = async function (context) {
                try {
                  runtime.context = context;
                  await loadTokenAndRouting(context);
                  await registerPanelOnce(context);
                } catch (_) { }
                return orig.apply(this, arguments);
              };
              cache.set(orig, wrapped);
              return wrapped;
            }
            return Reflect.get(target, prop, receiver);
          }
        });
      } catch (_) {
        return exp;
      }
    };

    let currentExports = wrapExports(moduleRef.exports);
    try { moduleRef.exports = currentExports; } catch (_) { }
    try {
      Object.defineProperty(moduleRef, "exports", {
        configurable: true,
        get() { return currentExports; },
        set(v) { try { currentExports = wrapExports(v); } catch (_) { currentExports = v; } }
      });
    } catch (_) { }
  }

  async function registerPanelOnce(context) {
    const runtime = ensureRuntime();
    const vscode = tryRequireVscode();
    if (!runtime || !context || !vscode || !vscode.commands || !vscode.window) return;
    if (runtime.__panelRegistered) return;
    runtime.__panelRegistered = true;

    const disposable = vscode.commands.registerCommand(COMMAND_ID, async () => {
      try {
        const initial = readAugmentAdvancedConfig(vscode);
        const panel = vscode.window.createWebviewPanel(VIEW_TYPE, TITLE, vscode.ViewColumn.Active, { enableScripts: true, retainContextWhenHidden: true });
        panel.webview.html = buildHtml(initial);
        panel.webview.onDidReceiveMessage(async (msg) => {
          try {
            if (!msg || typeof msg !== "object") return;
            if (msg.type === "openExternal" && typeof msg.url === "string" && msg.url.trim()) {
              return vscode.env.openExternal(vscode.Uri.parse(msg.url.trim()));
            }
            if (msg.type !== "rpc") return;
            const method = normalizeString(msg.method);
            const params = msg.params && typeof msg.params === "object" ? msg.params : {};
            const completionURL = normalizeString(readAugmentAdvancedConfig(vscode).completionURL);
            if (!completionURL && method !== "openSettings") throw new Error("augment.advanced.completionURL 为空");

            if (method === "getState") {
              const rules = runtime.routing?.rules && typeof runtime.routing?.rules === "object" ? runtime.routing.rules : {};
              const endpoints = Array.from(new Set([...IMPLEMENTED_ENDPOINTS.map(normalizeEndpoint), ...Object.keys(rules).map(normalizeEndpoint)])).filter(Boolean).sort((a, b) => a.localeCompare(b));
              let historySummary = { enabled: false, model: "" };
              try {
                const proxyCfg = await fetchProxyConfig({ completionURL });
                historySummary = extractHistorySummaryState(proxyCfg);
              } catch (_) { }
              panel.webview.postMessage({ type: "state", data: { rules, endpoints, models: runtime.models || [], historySummary } });
              return;
            }
            if (method === "openAdmin") {
              const url = joinBaseUrl(completionURL, "admin");
              if (!url) throw new Error("admin URL 构建失败（completionURL 无效）");
              return vscode.env.openExternal(vscode.Uri.parse(url));
            }
            if (method === "openSettings") {
              try { await vscode.commands.executeCommand("workbench.action.openSettings", "augment.advanced"); } catch (_) { await vscode.commands.executeCommand("workbench.action.openSettings"); }
              return;
            }
            if (method === "saveRouting") {
              await saveRouting(context, { version: 1, rules: params.rules });
              try { vscode.window.showInformationMessage("BYOK Proxy 路由规则已保存"); } catch (_) { }
              return;
            }
            if (method === "clearRouting") {
              await clearRouting(context);
              try { vscode.window.showInformationMessage("BYOK Proxy 路由规则已清空"); } catch (_) { }
              return;
            }
            if (method === "refreshModels") {
              const models = await fetchByokModels({ completionURL });
              runtime.models = models;
              const rules = runtime.routing?.rules && typeof runtime.routing?.rules === "object" ? runtime.routing.rules : {};
              const endpoints = Array.from(new Set([...IMPLEMENTED_ENDPOINTS.map(normalizeEndpoint), ...Object.keys(rules).map(normalizeEndpoint)])).filter(Boolean).sort((a, b) => a.localeCompare(b));
              let historySummary = { enabled: false, model: "" };
              try {
                const proxyCfg = await fetchProxyConfig({ completionURL });
                historySummary = extractHistorySummaryState(proxyCfg);
              } catch (_) { }
              return panel.webview.postMessage({ type: "state", data: { rules, endpoints, models, historySummary } });
            }

            if (method === "refreshHistorySummary") {
              const rules = runtime.routing?.rules && typeof runtime.routing?.rules === "object" ? runtime.routing.rules : {};
              const endpoints = Array.from(new Set([...IMPLEMENTED_ENDPOINTS.map(normalizeEndpoint), ...Object.keys(rules).map(normalizeEndpoint)])).filter(Boolean).sort((a, b) => a.localeCompare(b));
              const models = runtime.models || [];
              const proxyCfg = await fetchProxyConfig({ completionURL });
              const historySummary = extractHistorySummaryState(proxyCfg);
              return panel.webview.postMessage({ type: "state", data: { rules, endpoints, models, historySummary, historySummaryStatus: { ok: true } } });
            }

            if (method === "applyHistorySummary") {
              const enabled = Boolean(params.enabled);
              const model = normalizeString(params.model);
              const result = await applyProxyHistorySummary({ completionURL, enabled, model });
              try { vscode.window.showInformationMessage("History Summary 已应用到 Proxy（热更新）"); } catch (_) { }

              const rules = runtime.routing?.rules && typeof runtime.routing?.rules === "object" ? runtime.routing.rules : {};
              const endpoints = Array.from(new Set([...IMPLEMENTED_ENDPOINTS.map(normalizeEndpoint), ...Object.keys(rules).map(normalizeEndpoint)])).filter(Boolean).sort((a, b) => a.localeCompare(b));
              const models = runtime.models || [];
              let historySummary = { enabled: false, model: "" };
              try {
                const proxyCfg = await fetchProxyConfig({ completionURL });
                historySummary = extractHistorySummaryState(proxyCfg);
              } catch (_) { }
              return panel.webview.postMessage({ type: "state", data: { rules, endpoints, models, historySummary, historySummaryStatus: result || { ok: true } } });
            }

            if (method === "saveProxyConfig") {
              const result = await saveProxyConfigToFile({ completionURL });
              try { vscode.window.showInformationMessage("Proxy 配置已保存到 config.yaml"); } catch (_) { }
              const rules = runtime.routing?.rules && typeof runtime.routing?.rules === "object" ? runtime.routing.rules : {};
              const endpoints = Array.from(new Set([...IMPLEMENTED_ENDPOINTS.map(normalizeEndpoint), ...Object.keys(rules).map(normalizeEndpoint)])).filter(Boolean).sort((a, b) => a.localeCompare(b));
              const models = runtime.models || [];
              let historySummary = { enabled: false, model: "" };
              try {
                const proxyCfg = await fetchProxyConfig({ completionURL });
                historySummary = extractHistorySummaryState(proxyCfg);
              } catch (_) { }
              return panel.webview.postMessage({ type: "state", data: { rules, endpoints, models, historySummary, historySummaryStatus: result || { ok: true } } });
            }

            if (method === "clearHistorySummaryCache") {
              const result = await clearProxyHistorySummaryCacheAll({ completionURL });
              try { vscode.window.showInformationMessage("History Summary 缓存已清空"); } catch (_) { }
              const rules = runtime.routing?.rules && typeof runtime.routing?.rules === "object" ? runtime.routing.rules : {};
              const endpoints = Array.from(new Set([...IMPLEMENTED_ENDPOINTS.map(normalizeEndpoint), ...Object.keys(rules).map(normalizeEndpoint)])).filter(Boolean).sort((a, b) => a.localeCompare(b));
              const models = runtime.models || [];
              let historySummary = { enabled: false, model: "" };
              try {
                const proxyCfg = await fetchProxyConfig({ completionURL });
                historySummary = extractHistorySummaryState(proxyCfg);
              } catch (_) { }
              return panel.webview.postMessage({ type: "state", data: { rules, endpoints, models, historySummary, historySummaryStatus: result || { ok: true } } });
            }
          } catch (e) {
            try { vscode.window.showErrorMessage(`BYOK Proxy: ${String(e && e.message ? e.message : e)}`); } catch (_) { }
          }
        });
      } catch (e) {
        try { vscode.window.showErrorMessage(`BYOK Proxy Panel 打开失败: ${String(e && e.message ? e.message : e)}`); } catch (_) { }
      }
    });

    try { context.subscriptions.push(disposable); } catch (_) { }
  }

  try { installContextHook(); } catch (_) { }
})();
