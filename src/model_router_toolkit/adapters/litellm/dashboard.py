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
      <div class="label">Tokens (in / out)</div>
      <div class="val" id="tokens" style="font-size:20px">—</div>
    </div>
  </div>

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
function money(n){ return "$" + Number(n).toFixed(Number(n) < 1 ? 4 : 2); }
function fmt(n){ return Number(n).toLocaleString(); }

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
    $("tokens").textContent = fmt(d.input_tokens||0) + " / " + fmt(d.output_tokens||0);
    $("baseline_model").textContent = d.baseline_model || "—";
    $("uptime").textContent = Math.round(d.uptime_seconds||0) + "s";

    const dist = d.routing_distribution || {};
    const names = Object.keys(dist);
    const box = $("dist");
    if(!names.length){ box.innerHTML = '<div class="empty">No requests yet.</div>'; return; }
    box.innerHTML = names.map(n => {
      const e = dist[n];
      return `<div class="row">
        <div class="name" title="${n}">${n}</div>
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
