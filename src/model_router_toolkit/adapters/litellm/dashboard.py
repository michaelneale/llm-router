"""Self-contained HTML dashboard for the savings tracker.

Served at GET /dashboard. Polls /savings every few seconds, renders the
running totals + routing distribution, and offers a reset button.
No external dependencies (no CDN) so it works offline.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Router — Savings</title>
<style>
  :root {
    --bg:#0d1117; --card:#161b22; --line:#30363d; --fg:#e6edf3;
    --muted:#8b949e; --green:#3fb950; --accent:#58a6ff; --warn:#d29922;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--fg); padding:24px; }
  h1 { font-size:18px; font-weight:600; margin:0 0 2px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
          gap:14px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px 18px; }
  .card .label { color:var(--muted); font-size:12px; text-transform:uppercase;
                 letter-spacing:.04em; }
  .card .val { font-size:26px; font-weight:600; margin-top:6px; }
  .card .val.big { font-size:40px; }
  .green { color:var(--green); }
  .accent { color:var(--accent); }
  .hero { background:linear-gradient(135deg,#0d2818,#161b22); border-color:#1f6f3f; }
  section { background:var(--card); border:1px solid var(--line); border-radius:10px;
            padding:16px 18px; margin-bottom:18px; }
  section h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em;
               color:var(--muted); margin:0 0 14px; font-weight:600; }
  .row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .row .name { width:230px; font-size:13px; white-space:nowrap; overflow:hidden;
               text-overflow:ellipsis; }
  .bar { flex:1; background:#21262d; border-radius:5px; height:20px; overflow:hidden; }
  .bar > div { height:100%; background:linear-gradient(90deg,#1f6f3f,var(--green));
               border-radius:5px; transition:width .4s ease; }
  .row .pct { width:120px; text-align:right; color:var(--muted); font-size:12px;
              font-variant-numeric:tabular-nums; }
  .baseline { font-size:12px; color:var(--muted); }
  .knob-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
               gap:10px; margin-bottom:14px; }
  .knob { background:#0d1117; border:1px solid var(--line); border-radius:8px;
          padding:10px 12px; min-height:72px; }
  .knob .key { color:var(--muted); font-size:11px; text-transform:uppercase;
               letter-spacing:.04em; }
  .knob .value { font-size:18px; font-weight:600; margin-top:4px; }
  .knob .note { color:var(--muted); font-size:12px; margin-top:2px;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .pillrow { display:flex; gap:6px; flex-wrap:wrap; margin:4px 0 14px; }
  .pill { border:1px solid var(--line); border-radius:999px; padding:2px 8px;
          color:var(--muted); font-size:12px; background:#0d1117; }
  .pill.on { color:var(--green); border-color:#1f6f3f; }
  .pill.warn { color:var(--warn); border-color:#8a6a1f; }
  .tolerance-panel { background:#0d1117; border:1px solid var(--line);
                     border-radius:8px; padding:12px; margin-bottom:14px; }
  .tol-head { display:flex; align-items:baseline; justify-content:space-between;
              gap:10px; margin-bottom:8px; }
  .tol-head strong { font-size:22px; font-weight:600; }
  .tol-title { color:var(--muted); font-size:12px; text-transform:uppercase;
               letter-spacing:.04em; }
  .tol-source { color:var(--muted); font-size:12px; white-space:nowrap; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  .tol-scale { display:flex; justify-content:space-between; color:var(--muted);
               font-size:11px; margin-top:1px; }
  .presetrow { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  button.preset { padding:7px 10px; font-size:12px; }
  button.preset.active { color:var(--green); border-color:#1f6f3f; }
  button.preset .preset-note { display:block; color:var(--muted); font-size:10px;
                               margin-top:1px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { border-top:1px solid var(--line); padding:8px 6px; text-align:left;
          vertical-align:top; }
  th { color:var(--muted); font-weight:600; text-transform:uppercase;
       letter-spacing:.04em; font-size:11px; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .model-name { font-weight:600; }
  .model-slot { color:var(--muted); font-size:11px; margin-top:1px; }
  button { background:#21262d; color:var(--fg); border:1px solid var(--line);
           border-radius:8px; padding:9px 16px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); color:var(--accent); }
  button.danger:hover { border-color:#f85149; color:#f85149; }
  .toolbar { display:flex; gap:10px; align-items:center; }
  .dot { width:8px;height:8px;border-radius:50%;background:var(--green);
         display:inline-block;margin-right:6px; }
  .muted { color:var(--muted); }
  .empty { color:var(--muted); font-size:13px; padding:6px 0; }
  code { background:#21262d; padding:1px 6px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
  <h1>LLM Router — Live Savings</h1>
  <div class="sub">
    <span class="dot"></span><span id="status">connecting…</span>
    &nbsp;·&nbsp; routing locally, comparing every request against the baseline tier.
  </div>

  <div class="grid">
    <div class="card hero">
      <div class="label">Saved vs baseline</div>
      <div class="val big green" id="saved_pct">—</div>
    </div>
    <div class="card">
      <div class="label">Dollars saved</div>
      <div class="val green" id="saved_usd">—</div>
    </div>
    <div class="card">
      <div class="label">Actual spend</div>
      <div class="val" id="actual">—</div>
    </div>
    <div class="card">
      <div class="label">Baseline would cost</div>
      <div class="val muted" id="baseline">—</div>
    </div>
    <div class="card">
      <div class="label">Requests</div>
      <div class="val accent" id="requests">—</div>
    </div>
    <div class="card">
      <div class="label">Tokens (in / cached / out)</div>
      <div class="val" id="tokens" style="font-size:20px">—</div>
    </div>
  </div>

  <section>
    <h2>Routing knobs</h2>
    <div id="knobs"><div class="empty">Waiting for router config.</div></div>
  </section>

  <section>
    <h2>Routing distribution</h2>
    <div id="dist"><div class="empty">No requests yet.</div></div>
  </section>

  <section>
    <div class="toolbar">
      <button id="refresh">Refresh now</button>
      <button id="reset" class="danger">Reset counters</button>
      <span class="baseline">baseline: <code id="baseline_model">—</code>
        &nbsp;·&nbsp; uptime <span id="uptime">—</span></span>
    </div>
  </section>

  <div class="baseline">
    Auto-refreshes every 3s. Streaming and non-streaming requests are both counted.
    Tracks cost, not answer quality.
  </div>

<script>
const $ = id => document.getElementById(id);
let toleranceSaving = false;
function money(n){ return "$" + Number(n).toFixed(Number(n) < 1 ? 4 : 2); }
function fmt(n){ return Number(n).toLocaleString(); }
function esc(v){
  return String(v ?? "").replace(/[&<>"']/g, c => {
    if(c === "&") return "&amp;";
    if(c === "<") return "&lt;";
    if(c === ">") return "&gt;";
    if(c === '"') return "&quot;";
    return "&#39;";
  });
}

async function setTolerance(value){
  const tolerance = Math.max(0, Math.min(1, Number(value)));
  toleranceSaving = true;
  $("status").textContent = "updating tolerance…";
  try {
    const r = await fetch("/router/tuning", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({tolerance})
    });
    if(!r.ok) throw new Error(await r.text());
    const d = await r.json();
    renderKnobs(d.routing_knobs || {});
    $("status").textContent = "live";
  } catch(e) {
    $("status").textContent = "tolerance update failed";
  } finally {
    toleranceSaving = false;
  }
}

function renderKnobs(k){
  const box = $("knobs");
  if(!k || !Object.keys(k).length){
    box.innerHTML = '<div class="empty">No router config exposed yet.</div>';
    return;
  }
  if(k.error){
    box.innerHTML = `<div class="empty">${esc(k.error)}</div>`;
    return;
  }

  const sw = k.switching || {};
  const top = k.top_tier || {};
  const override = k.manual_override || "none";
  const tolerance = Number(k.effective_tolerance || 0);
  const startup = Number(k.startup_tolerance ?? k.configured_tolerance ?? tolerance);
  const source = k.tolerance_source || (k.env_tolerance ? "env" : "config");
  const tolNote = source === "runtime"
    ? `runtime · startup ${startup.toFixed(3)}`
    : `${source} default`;
  const switchState = sw.effective ? "on" : "off";
  const switchNote = sw.effective
    ? `up ${Number(sw.up_margin || 0).toFixed(2)} · down ${Number(sw.down_margin || 0).toFixed(2)} + ${Number(sw.down_margin_per_100k || 0).toFixed(2)}/100k`
    : (sw.disabled_by_env ? "disabled by env" : "disabled in config");
  const models = Array.isArray(k.models) ? k.models : [];
  const presets = Array.isArray(k.recommended_tolerances) ? k.recommended_tolerances : [];
  const presetButtons = presets.map(p => {
    const value = Number(p.value || 0);
    const active = Math.abs(value - tolerance) < 0.0006;
    return `<button class="preset ${active ? "active" : ""}" data-tolerance="${value.toFixed(3)}">
      ${esc(p.label || value.toFixed(3))} ${value.toFixed(3)}
      <span class="preset-note">${esc(p.note || "")}</span>
    </button>`;
  }).join("");
  const rows = models.map(m => `
    <tr>
      <td>
        <div class="model-name">${esc(m.display_name || m.slot)}</div>
        <div class="model-slot">${esc(m.slot)}</div>
      </td>
      <td>${esc(m.litellm_model || "")}</td>
      <td class="num">${Number(m.input_cost || 0).toFixed(2)} / ${Number(m.output_cost || 0).toFixed(2)}</td>
      <td class="num">${Number(m.routing_blend || 0).toFixed(3)}${Number(m.routing_cost_multiplier || 1) !== 1 ? ` ×${Number(m.routing_cost_multiplier).toFixed(2)}` : ""}</td>
    </tr>`).join("");

  box.innerHTML = `
    <div class="tolerance-panel">
      <div class="tol-head">
        <div>
          <div class="tol-title">Runtime tolerance</div>
          <strong id="tol_readout">${tolerance.toFixed(3)}</strong>
        </div>
        <div class="tol-source">${esc(tolNote)}</div>
      </div>
      <input id="tol_slider" type="range" min="0.020" max="0.220" step="0.005" value="${tolerance.toFixed(3)}">
      <div class="tol-scale"><span>quality</span><span>cheaper</span></div>
      <div class="presetrow">${presetButtons}</div>
    </div>
    <div class="knob-grid">
      <div class="knob">
        <div class="key">Tolerance</div>
        <div class="value">${tolerance.toFixed(3)}</div>
        <div class="note">${esc(tolNote)}</div>
      </div>
      <div class="knob">
        <div class="key">Cost key</div>
        <div class="value">in + ${Number(k.output_token_weight || 0).toFixed(2)}×out</div>
        <div class="note">sorted cheapest first</div>
      </div>
      <div class="knob">
        <div class="key">Switching</div>
        <div class="value">${esc(switchState)}</div>
        <div class="note" title="${esc(switchNote)}">${esc(switchNote)}</div>
      </div>
      <div class="knob">
        <div class="key">Top tier</div>
        <div class="value">${esc(top.display_name || "none")}</div>
        <div class="note">${esc(top.litellm_model || "")}</div>
      </div>
      <div class="knob">
        <div class="key">Manual top</div>
        <div class="value">${esc(override)}</div>
        <div class="note">explicit frontier override</div>
      </div>
      <div class="knob">
        <div class="key">Route log</div>
        <div class="value">${k.route_log ? "on" : "off"}</div>
        <div class="note" title="${esc(k.route_log || "")}">${esc(k.route_log || "not configured")}</div>
      </div>
    </div>
    <div class="pillrow">
      <span class="pill ${sw.effective ? "on" : "warn"}">switching ${esc(switchState)}</span>
      <span class="pill">pool ${esc((k.pool_config || "").split("/").pop() || "")}</span>
      <span class="pill">proxy ${esc((k.litellm_config || "").split("/").pop() || "")}</span>
    </div>
    <table>
      <thead><tr><th>Routing ladder</th><th>Provider model</th><th class="num">$/M in / out</th><th class="num">Blend</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4" class="muted">No models loaded.</td></tr>'}</tbody>
    </table>`;

  const slider = $("tol_slider");
  const readout = $("tol_readout");
  if(slider && readout){
    slider.oninput = () => { readout.textContent = Number(slider.value).toFixed(3); };
    slider.onchange = () => setTolerance(slider.value);
  }
  document.querySelectorAll("button.preset[data-tolerance]").forEach(btn => {
    btn.onclick = () => setTolerance(btn.dataset.tolerance);
  });
}

async function load(){
  try {
    const r = await fetch("/savings", {cache:"no-store"});
    const d = await r.json();
    $("status").textContent = "live";
    $("saved_pct").textContent = (d.saved_pct ?? 0).toFixed(1) + "%";
    $("saved_usd").textContent = money(d.saved_usd || 0);
    $("actual").textContent = money(d.actual_cost_usd || 0);
    $("baseline").textContent = money(d.baseline_cost_usd || 0);
    $("requests").textContent = fmt(d.requests || 0);
    $("tokens").textContent = fmt(d.input_tokens||0) + " / " +
      fmt(d.cached_input_tokens||0) + " / " + fmt(d.output_tokens||0);
    $("baseline_model").textContent = d.baseline_model || "—";
    $("uptime").textContent = Math.round(d.uptime_seconds||0) + "s";
    if(!toleranceSaving) renderKnobs(d.routing_knobs || {});

    const dist = d.routing_distribution || {};
    const names = Object.keys(dist);
    const box = $("dist");
    if(!names.length){ box.innerHTML = '<div class="empty">No requests yet.</div>'; return; }
    box.innerHTML = names.map(n => {
      const e = dist[n];
      return `<div class="row">
        <div class="name" title="${esc(n)}">${esc(n)}</div>
        <div class="bar"><div style="width:${e.share_pct}%"></div></div>
        <div class="pct">${e.share_pct}% · ${e.requests} · ${money(e.actual_cost_usd)}</div>
      </div>`;
    }).join("");
  } catch(e){
    $("status").textContent = "proxy unreachable";
  }
}

$("refresh").onclick = load;
$("reset").onclick = async () => {
  if(!confirm("Reset all savings counters to zero?")) return;
  await fetch("/savings/reset", {method:"POST"});
  load();
};
load();
setInterval(load, 3000);
</script>
</body>
</html>
"""
