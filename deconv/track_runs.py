"""Live loss tracker for training runs: reads the `step ...` lines train_cpu.py prints.

Pulls each run's log over ssh (or reads a local file), and writes one self-contained HTML
page with training loss, validation NLL and the latest kept-type RMSE per regime. The page
reloads itself every minute. Nothing leaves this machine.

  py -3.11 -B track_runs.py --host as22836@<ip> --every 120 \
      --run "v6 (cohort input)=deconv_v6/log_v6.txt" --run "v6 no cohort=deconv_v6/log_v6_nocohort.txt"

Without --host the paths are local files. --every 0 writes the page once and exits.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import re
import subprocess
import time
from pathlib import Path

LINE = re.compile(r"^step (\d+) loss (-?[\d.]+)(?: val_nll (-?[\d.]+) kept_rmse (\{.*?\}))? (\d+)s")
COLORS = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]


def read(host, path):
    if host:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, f"grep '^step' {path}"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode not in (0, 1):
            raise RuntimeError(out.stderr.strip()[:200])
        return out.stdout
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse(text):
    train, evals = {}, []
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        step, loss, seconds = int(m[1]), float(m[2]), int(m[5])
        if m[3] is None:
            train[step] = dict(step=step, loss=loss, seconds=seconds)
        else:
            evals.append(dict(step=step, val_nll=float(m[3]), rmse=ast.literal_eval(m[4]), seconds=seconds))
            train.setdefault(step, dict(step=step, loss=loss, seconds=seconds))
    return sorted(train.values(), key=lambda r: r["step"]), evals


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Training tracker</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7}
@media (prefers-color-scheme: dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835}}
body{margin:0;padding:20px 16px 40px;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto}
h1{font-size:20px;margin:0 0 2px} .sub{color:var(--ink2);margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-bottom:16px}
.tile,.card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums} .tile span{color:var(--ink2)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.card h2{font-size:14px;margin:0 0 2px} .card p{margin:0 0 6px;color:var(--ink2);font-size:12px}
svg{width:100%;height:auto;display:block;overflow:visible} svg text{fill:var(--muted);font-size:11px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:4px 0 12px;color:var(--ink2)}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:8px;padding:6px 9px;font-size:12px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15);font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto} table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap} th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
</style></head><body><main>
<h1>Training tracker</h1><p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div><div class="legend" id="legend"></div>
<div class="charts">
 <div class="card"><h2>Training loss</h2><p>Mean negative log-likelihood over the last 100 steps (lower is better)</p><div id="c1"></div></div>
 <div class="card"><h2>Validation NLL</h2><p>Held-out-donor synthetic cohorts, every evaluation (lower is better)</p><div id="c2"></div></div>
</div>
<div class="card" style="margin-top:12px"><h2>Kept-type RMSE at the latest evaluation</h2><p>Percentage points, synthetic validation, by regime</p><div class="scroll" id="tbl"></div></div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const col = i => D.colors[i % D.colors.length][dark ? 1 : 0];
const fmt = (x, d=2) => x == null ? '–' : x.toFixed(d);
document.getElementById('sub').textContent = 'Updated ' + D.updated + (D.note ? ' · ' + D.note : '') + ' · reloads every minute';
document.getElementById('legend').innerHTML = D.runs.map((r,i)=>`<span><i class="dot" style="background:${col(i)}"></i>${r.name}</span>`).join('');
document.getElementById('tiles').innerHTML = D.runs.map((r,i)=>{
  const t = r.train.at(-1), e = r.evals.at(-1);
  if(!t) return `<div class="tile"><span><i class="dot" style="background:${col(i)}"></i>${r.name}</span><b>no data yet</b></div>`;
  const rate = t.seconds / t.step, left = Math.max(0, (D.total - t.step) * rate);
  const eta = left ? Math.floor(left/3600) + 'h ' + Math.round(left%3600/60) + 'm left' : 'finished';
  return `<div class="tile"><span><i class="dot" style="background:${col(i)}"></i>${r.name}</span><b>step ${t.step.toLocaleString()} / ${D.total.toLocaleString()}</b>
   <span>loss ${fmt(t.loss)} · val NLL ${e ? fmt(e.val_nll) : '–'} · ${fmt(rate,2)} s/step · ${eta}</span></div>`}).join('');
function chart(id, key, src){
  const W=520,H=280,L=46,R=12,T=10,B=28, el=document.getElementById(id);
  const series = D.runs.map(r => r[src].filter(p => p[key] != null));
  const all = series.flat(); if(!all.length){ el.textContent='No data yet'; return; }
  const xmax = Math.max(D.total, ...all.map(p=>p.step));
  let ys = all.map(p=>p[key]).sort((a,b)=>a-b);
  let lo = ys[0], hi = ys[Math.min(ys.length-1, Math.floor(ys.length*0.97))];   // clip the start-up spike
  if(hi - lo < 1e-9) hi = lo + 1; const pad=(hi-lo)*0.06; lo-=pad; hi+=pad;
  const X = s => L + (W-L-R)*s/xmax, Y = v => T + (H-T-B)*(1-(Math.min(v,hi)-lo)/(hi-lo));
  let g = '';
  for(let k=0;k<=4;k++){ const v=lo+(hi-lo)*k/4, y=Y(v); g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="${L-6}" y="${y+4}" text-anchor="end">${v.toFixed(1)}</text>`; }
  for(let k=0;k<=4;k++){ const s=xmax*k/4; g+=`<text x="${X(s)}" y="${H-8}" text-anchor="middle">${(s/1000).toFixed(0)}k</text>`; }
  g+=`<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="var(--axis)"/>`;
  series.forEach((s,i)=>{ if(!s.length) return;
    g+=`<polyline fill="none" stroke="${col(i)}" stroke-width="2" stroke-linejoin="round" points="${s.map(p=>X(p.step)+','+Y(p[key])).join(' ')}"/>`;
    if(s.length<60) s.forEach(p=>g+=`<circle cx="${X(p.step)}" cy="${Y(p[key])}" r="3" fill="${col(i)}" stroke="var(--surface)" stroke-width="1.5"/>`); });
  g+=`<line id="${id}x" y1="${T}" y2="${H-B}" stroke="var(--axis)" visibility="hidden"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${key} by step">${g}</svg>`;
  const svg = el.querySelector('svg'), tip = document.getElementById('tip'), cross = document.getElementById(id+'x');
  svg.addEventListener('mousemove', ev => {
    const r = svg.getBoundingClientRect(), step = (ev.clientX - r.left) / r.width * W;
    const s0 = Math.max(0, (step - L) / (W-L-R) * xmax);
    const rows = series.map((s,i)=>{ if(!s.length) return ''; const p = s.reduce((a,b)=>Math.abs(b.step-s0)<Math.abs(a.step-s0)?b:a);
      return `<div><i class="dot" style="background:${col(i)}"></i>${D.runs[i].name}: <b>${fmt(p[key])}</b> at step ${p.step.toLocaleString()}</div>`}).join('');
    cross.setAttribute('x1', X(s0)); cross.setAttribute('x2', X(s0)); cross.setAttribute('visibility','visible');
    tip.innerHTML = rows; tip.style.display='block';
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - 300) + 'px'; tip.style.top = (ev.clientY + 14) + 'px'; });
  svg.addEventListener('mouseleave', ()=>{ tip.style.display='none'; cross.setAttribute('visibility','hidden'); });
}
chart('c1','loss','train'); chart('c2','val_nll','evals');
const regimes = [...new Set(D.runs.flatMap(r => r.evals.length ? Object.keys(r.evals.at(-1).rmse) : []))];
document.getElementById('tbl').innerHTML = regimes.length ? `<table><tr><th>Run</th><th>Step</th>${regimes.map(x=>`<th>${x}</th>`).join('')}</tr>` +
  D.runs.map((r,i)=>{ const e=r.evals.at(-1); return `<tr><td><i class="dot" style="background:${col(i)}"></i>${r.name}</td><td>${e?e.step.toLocaleString():'–'}</td>${regimes.map(x=>`<td>${e?fmt(e.rmse[x]):'–'}</td>`).join('')}</tr>`}).join('') + '</table>' : 'No evaluation yet';
</script></main></body></html>
"""


def write(runs, total, out, note=""):
    data = dict(runs=runs, total=total, colors=COLORS, note=note,
                updated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    out.write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=path to the log")
    ap.add_argument("--host", default=None, help="ssh target; omit for local log files")
    ap.add_argument("--total", type=int, default=40000, help="planned number of steps")
    ap.add_argument("--every", type=int, default=120, help="seconds between refreshes; 0 = once")
    ap.add_argument("--out", type=Path, default=Path("results/training_tracker.html"))
    args = ap.parse_args()
    specs = [r.split("=", 1) for r in args.run]
    last = {name: ([], []) for name, _ in specs}
    while True:
        note = ""
        for name, path in specs:
            try:
                last[name] = parse(read(args.host, path))
            except Exception as exc:  # keep the last good data; say so on the page
                note = f"last refresh failed ({type(exc).__name__}); showing earlier data"
        runs = [dict(name=name, train=last[name][0], evals=last[name][1]) for name, _ in specs]
        write(runs, args.total, args.out, note)
        done = all(r["train"] and r["train"][-1]["step"] >= args.total for r in runs)
        if args.every <= 0 or done:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
