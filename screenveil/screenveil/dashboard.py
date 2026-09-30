"""Local-only live dashboard (http://127.0.0.1:8765). Standard library only.

Binds to localhost and serves telemetry only — never frames or text — so
nothing sensitive can leave the machine through it.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ScreenVeil Conductor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaf0;--mute:#8b93a7;--acc:#3ddc97;--warn:#ffb020;--npu:#3ddc97;--gpu:#6aa7ff;--cpu:#ff7a59}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}
.card{background:var(--card);border-radius:12px;padding:14px}
.k{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:clamp(17px,1.9vw,24px);font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:700px){.row{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse}td,th{padding:6px 4px;text-align:left;border-bottom:1px solid #262a33}
.pill{padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;color:#0f1115}
.NPU{background:var(--npu)}.GPU{background:var(--gpu)}.CPU{background:var(--cpu)}
.ev{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--mute)}
.note{color:var(--mute);font-size:12px;margin-top:8px}
</style></head><body><main>
<h1>ScreenVeil · NPU Conductor</h1><div class="sub" id="sub">connecting…</div>
<div class="grid">
 <div class="card"><div class="k">Power mode</div><div class="v" id="mode">–</div></div>
 <div class="card"><div class="k">Battery</div><div class="v" id="bat">–</div></div>
 <div class="card"><div class="k">FPS / target</div><div class="v" id="fps">–</div></div>
 <div class="card"><div class="k">Active redactions</div><div class="v" id="red">–</div></div>
 <div class="card"><div class="k">Unverified screen</div><div class="v" id="pend">–</div></div>
 <div class="card"><div class="k">Energy vs CPU-only (est.)</div><div class="v" id="en">–</div></div>
</div>
<div class="row">
 <div class="card"><div class="k">Where each model runs</div><table id="wl"></table></div>
 <div class="card"><div class="k">Detections by type</div><table id="kinds"></table></div>
</div>
<div class="card" style="margin-top:12px"><div class="k">Conductor decisions</div><div id="ev" class="ev"></div>
<div class="note">Energy is estimated from measured latency × per-device power coefficients; measured battery discharge is shown when Windows reports it.</div></div>
</main><script>
async function tick(){try{const s=await (await fetch('/stats')).json();
const $=id=>document.getElementById(id);
$('sub').textContent=(s.healthy?'Protecting':'FAIL-CLOSED: shield active')+' · frame '+s.frames+' · last frame '+(s.last_frame.total_ms||0)+' ms';
$('mode').textContent=s.mode;$('bat').textContent=s.battery_percent==null?'AC':Math.round(s.battery_percent)+'%'+(s.on_ac?' ⚡':'')+(s.measured_discharge_w?' · '+s.measured_discharge_w.toFixed(1)+' W':'');
$('fps').textContent=s.fps+' / '+s.target_fps;$('red').textContent=s.active_redactions;$('pend').textContent=s.pending_pct+'%';
$('en').textContent=s.energy_saved_pct_est>0?'−'+s.energy_saved_pct_est+'%':'—';
$('wl').innerHTML='<tr><th>Model</th><th>Device</th><th>Latency (ms)</th></tr>'+Object.entries(s.workloads).map(([k,w])=>'<tr><td>'+k+'</td><td><span class="pill '+w.device+'">'+w.device+'</span></td><td>'+Object.entries(w.latency_ms).map(([d,v])=>d+' '+v).join(' · ')+'</td></tr>').join('');
$('kinds').innerHTML=Object.entries(s.detections_by_kind).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<tr><td>'+k+'</td><td>'+v+'</td></tr>').join('')||'<tr><td>none yet</td></tr>';
$('ev').innerHTML=s.events.slice().reverse().map(e=>e[0]+'  '+e[1]).join('<br>');
}catch(e){document.getElementById('sub').textContent='dashboard disconnected'}}
setInterval(tick,700);tick();
</script></body></html>"""


def serve(snapshot: Callable[[], dict], port: int = 8765) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stats"):
                body, ctype = json.dumps(snapshot()).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
