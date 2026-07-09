#!/usr/bin/env python3
"""Benchmark dashboard — a zero-dependency web UI for db/benchmark.sqlite.

    uv run bench_server.py [--port 8765]      (plain python3 works too)
    ->  http://localhost:8765

Shows the fair PLN-vs-RND comparison: success rate, median total emulator
frames, wall-clock, real deaths, replayable rate — per stage, filterable by
fresh/cumulative condition, auto-refreshing while benchmarks run.
"""
import argparse
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BENCH_DB = os.path.join('db', 'benchmark.sqlite')

PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RandoMario — Fair Benchmark</title>
<style>
:root{
  --bg:#0d0f14; --bg2:#101218; --panel:#171b25; --panel2:#1e2330;
  --border:#2a3040; --text:#e2e8f0; --dim:#8b95a8;
  --rnd:#fbbf24; --pln:#4ade80; --blue:#60a5fa; --red:#f87171;
}
*{box-sizing:border-box}
body{margin:0;background:
  radial-gradient(900px 380px at 85% -10%,rgba(74,222,128,.07),transparent 60%),
  radial-gradient(900px 380px at 15% -10%,rgba(251,191,36,.07),transparent 60%),
  var(--bg);
  color:var(--text);font:15px/1.6 'Segoe UI','Hiragino Sans','Noto Sans JP',sans-serif;}
.wrap{max-width:1280px;margin:0 auto;padding:28px 24px 80px}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:24px;margin:0;letter-spacing:.02em}
h1 .mush{margin-right:8px}
.sub{color:var(--dim);font-size:13px}
.legend{margin-left:auto;display:flex;gap:14px;font-size:13px}
.legend b{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.chips{display:flex;gap:8px;margin:18px 0 22px;flex-wrap:wrap}
.chip{background:var(--panel);border:1px solid var(--border);color:var(--dim);
  border-radius:999px;padding:5px 16px;font-size:13px;cursor:pointer;user-select:none;
  transition:all .15s}
.chip:hover{border-color:var(--blue)}
.chip.on{background:var(--panel2);color:var(--text);border-color:var(--blue)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-bottom:26px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px 18px;position:relative;overflow:hidden}
.card .t{color:var(--dim);font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px}
.duel{display:flex;align-items:center;gap:10px}
.duel .v{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.duel .v.rnd{color:var(--rnd)} .duel .v.pln{color:var(--pln)}
.duel .vs{color:var(--dim);font-size:12px}
.ratio{position:absolute;top:14px;right:14px;background:rgba(74,222,128,.12);
  color:var(--pln);border-radius:8px;padding:2px 9px;font-size:12px;font-weight:700}
.ratio.neg{background:rgba(248,113,113,.12);color:var(--red)}
section{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  padding:20px 22px;margin-bottom:22px}
section h2{font-size:15px;color:var(--dim);letter-spacing:.06em;margin:0 0 16px;
  text-transform:uppercase}
.metric-chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
svg text{font:12px 'Consolas','Segoe UI',monospace;fill:var(--dim)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px;letter-spacing:.05em}
td.agent-rnd{color:var(--rnd);font-weight:700;text-align:left}
td.agent-pln{color:var(--pln);font-weight:700;text-align:left}
tr:last-child td{border-bottom:none}
.ok{color:var(--pln)} .bad{color:var(--red)} .mid{color:var(--rnd)}
.empty{color:var(--dim);text-align:center;padding:44px 0}
footer{color:var(--dim);font-size:12.5px;text-align:center;margin-top:8px}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--pln);
  margin-right:7px;animation:pu 2s infinite}
@keyframes pu{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="mush">🍄</span>RandoMario — Fair Benchmark</h1>
    <span class="sub"><span class="pulse"></span>auto-refresh · success × total emulator frames × wall-clock, fresh条件</span>
    <div class="legend">
      <span><b style="background:var(--rnd)"></b>RANDOM (Go-Explore)</span>
      <span><b style="background:var(--pln)"></b>PLANNER (model-based)</span>
    </div>
  </header>
  <div class="chips" id="modeChips"></div>
  <div class="cards" id="cards"></div>
  <section>
    <h2>ステージ別比較 <span id="chartMetricLabel"></span></h2>
    <div class="metric-chips" id="metricChips"></div>
    <div id="chart"></div>
  </section>
  <section>
    <h2>ステージ別 集計(中央値)</h2>
    <div id="tableWrap"></div>
  </section>
  <section>
    <h2>生データ(最新50ラン)</h2>
    <div id="rawWrap"></div>
  </section>
  <footer>db/benchmark.sqlite · <code>uv run benchmark.py --stages … --reps N --budget F</code> で追加計測</footer>
</div>
<script>
const METRICS = [
  ['total_frames','総エミュレータフレーム',true],
  ['real_frames','実プレイフレーム',true],
  ['wall_sec','wall-clock 秒',true],
  ['deaths_to_clear','実死亡数',false],
  ['clear_ep','クリアEP',false],
];
let MODE='fresh', METRIC='total_frames', DATA=[];

const fmt = n => n==null?'—':(n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e4?(n/1e3).toFixed(0)+'k':(Math.round(n*10)/10).toLocaleString());
const median = a => {if(!a.length)return null;const s=[...a].sort((x,y)=>x-y);const m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2;};

function agg(rows){
  const key = r=>r.stage+'|'+r.agent;
  const g = {};
  rows.forEach(r=>{(g[key(r)]=g[key(r)]||[]).push(r)});
  return Object.entries(g).map(([k,rs])=>{
    const [stage,agent]=k.split('|');
    const cleared=rs.filter(r=>r.cleared);
    const m=f=>median(cleared.map(r=>r[f]).filter(v=>v!=null));
    return {stage,agent,n:rs.length,ok:cleared.length,
      success:cleared.length/rs.length,
      clear_ep:m('clear_ep'),deaths_to_clear:m('deaths_to_clear'),
      real_frames:m('real_frames'),rollout_frames:m('rollout_frames'),
      total_frames:m('total_frames'),wall_sec:m('wall_sec'),
      replayable:cleared.length?cleared.filter(r=>r.replayable).length/cleared.length:null};
  });
}

function ratioBadge(rv,pv,lowerBetter=true){
  if(rv==null||pv==null||pv===0||rv===0)return '';
  const r = lowerBetter? rv/pv : pv/rv;
  const good = r>=1;
  return `<span class="ratio ${good?'':'neg'}">PLN ×${r>=10?r.toFixed(0):r.toFixed(1)}</span>`;
}

function renderCards(rows){
  const R=rows.filter(r=>r.agent==='random'), P=rows.filter(r=>r.agent==='planner');
  const rate=a=>{const n=a.reduce((s,r)=>s+r.n,0);return n?a.reduce((s,r)=>s+r.ok,0)/n:null};
  const med=(a,f)=>median(a.map(r=>r[f]).filter(v=>v!=null));
  const pct=v=>v==null?'—':(v*100).toFixed(0)+'%';
  const cards=[
    ['成功率', pct(rate(R)), pct(rate(P)), ''],
    ['総フレーム (中央値)', fmt(med(R,'total_frames')), fmt(med(P,'total_frames')),
      ratioBadge(med(R,'total_frames'),med(P,'total_frames'))],
    ['wall-clock (中央値)', fmt(med(R,'wall_sec'))+'s', fmt(med(P,'wall_sec'))+'s',
      ratioBadge(med(R,'wall_sec'),med(P,'wall_sec'))],
    ['実死亡数 (中央値)', fmt(med(R,'deaths_to_clear')), fmt(med(P,'deaths_to_clear')),
      ratioBadge(med(R,'deaths_to_clear'),med(P,'deaths_to_clear'))],
    ['再現可能率', pct(med(R,'replayable')), pct(med(P,'replayable')), ''],
  ];
  document.getElementById('cards').innerHTML = cards.map(([t,rv,pv,badge])=>`
    <div class="card">${badge}<div class="t">${t}</div>
      <div class="duel"><span class="v rnd">${rv}</span><span class="vs">RND / PLN</span>
      <span class="v pln">${pv}</span></div></div>`).join('');
}

function renderChart(rows){
  const stages=[...new Set(rows.map(r=>r.stage))].sort((a,b)=>{
    const [aw,as]=a.split('-').map(Number),[bw,bs]=b.split('-').map(Number);
    return aw-bw||as-bs;});
  const log = METRICS.find(m=>m[0]===METRIC)[2];
  const vals=rows.map(r=>r[METRIC]).filter(v=>v!=null&&v>0);
  if(!stages.length||!vals.length){document.getElementById('chart').innerHTML='<div class="empty">データなし — benchmark.py を実行してください</div>';return;}
  const max=Math.max(...vals), min=Math.min(...vals);
  const W=1180, rowH=46, H=stages.length*rowH+30, L=70, R=90;
  const sc = v=>{
    if(v==null||v<=0)return 0;
    if(log){const lo=Math.log10(Math.max(min,1)),hi=Math.log10(max);
      return hi===lo?W-L-R:(Math.log10(v)-lo)/(hi-lo)*(W-L-R)*0.92+(W-L-R)*0.08;}
    return v/max*(W-L-R);
  };
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  stages.forEach((st,i)=>{
    const y=i*rowH+22;
    const r=rows.find(x=>x.stage===st&&x.agent==='random');
    const p=rows.find(x=>x.stage===st&&x.agent==='planner');
    s+=`<text x="8" y="${y+16}" style="fill:var(--text);font-weight:700">${st}</text>`;
    [[r,'var(--rnd)',0],[p,'var(--pln)',17]].forEach(([row,col,dy])=>{
      const v=row?row[METRIC]:null;
      const w=v!=null?Math.max(sc(v),3):0;
      const fail=row&&row.success<1;
      s+=`<rect x="${L}" y="${y+dy}" width="${w}" height="13" rx="4" fill="${col}" opacity="${row&&row.success===0?0.25:0.92}"/>`;
      s+=`<text x="${L+w+8}" y="${y+dy+11}">${v!=null?fmt(v):(row?'not cleared':'—')}${row&&row.success<1&&row.success>0?` (${(row.success*100).toFixed(0)}%)`:''}</text>`;
    });
  });
  s+='</svg>';
  document.getElementById('chart').innerHTML=s;
}

function renderTable(rows){
  const stages=[...new Set(rows.map(r=>r.stage))].sort();
  if(!rows.length){document.getElementById('tableWrap').innerHTML='<div class="empty">データなし</div>';return;}
  let h=`<table><tr><th>stage / agent</th><th>runs</th><th>成功率</th><th>クリアEP</th>
    <th>実死亡</th><th>実フレーム</th><th>rolloutフレーム</th><th>総フレーム</th>
    <th>wall(s)</th><th>再現可</th></tr>`;
  stages.forEach(st=>{
    ['random','planner'].forEach(ag=>{
      const r=rows.find(x=>x.stage===st&&x.agent===ag);
      if(!r)return;
      const cls=ag==='random'?'agent-rnd':'agent-pln';
      const sr=r.success;
      h+=`<tr><td class="${cls}">${st} · ${ag==='random'?'RND':'PLN'}</td>
        <td>${r.ok}/${r.n}</td>
        <td class="${sr===1?'ok':sr>0?'mid':'bad'}">${(sr*100).toFixed(0)}%</td>
        <td>${fmt(r.clear_ep)}</td><td>${fmt(r.deaths_to_clear)}</td>
        <td>${fmt(r.real_frames)}</td><td>${fmt(r.rollout_frames)}</td>
        <td><b>${fmt(r.total_frames)}</b></td><td>${fmt(r.wall_sec)}</td>
        <td>${r.replayable==null?'—':(r.replayable*100).toFixed(0)+'%'}</td></tr>`;
    });
  });
  document.getElementById('tableWrap').innerHTML=h+'</table>';
}

function renderRaw(runs){
  const rs=[...runs].sort((a,b)=>b.created-a.created).slice(0,50);
  if(!rs.length){document.getElementById('rawWrap').innerHTML='<div class="empty">データなし</div>';return;}
  let h=`<table><tr><th>time</th><th>stage</th><th>agent</th><th>mode</th><th>rep</th>
    <th>clear</th><th>EP</th><th>死亡</th><th>実F</th><th>rolloutF</th><th>総F</th>
    <th>wall</th><th>再現</th></tr>`;
  rs.forEach(r=>{
    const t=new Date(r.created*1000).toLocaleTimeString();
    h+=`<tr><td>${t}</td><td>${r.stage}</td>
      <td class="${r.agent==='random'?'agent-rnd':'agent-pln'}">${r.agent==='random'?'RND':'PLN'}</td>
      <td>${r.mode}</td><td>${r.rep}</td>
      <td class="${r.cleared?'ok':'bad'}">${r.cleared?'✓':'✗'}</td>
      <td>${fmt(r.clear_ep)}</td><td>${fmt(r.deaths_to_clear)}</td>
      <td>${fmt(r.real_frames)}</td><td>${fmt(r.rollout_frames)}</td>
      <td>${fmt(r.total_frames)}</td><td>${r.wall_sec==null?'—':fmt(r.wall_sec)+'s'}</td>
      <td>${r.replayable==null?'—':r.replayable?'✓':'✗'}</td></tr>`;
  });
  document.getElementById('rawWrap').innerHTML=h+'</table>';
}

function renderChips(){
  const modes=[...new Set(DATA.map(r=>r.mode))];
  if(!modes.includes(MODE)&&modes.length)MODE=modes[0];
  document.getElementById('modeChips').innerHTML =
    modes.map(m=>`<span class="chip ${m===MODE?'on':''}" onclick="MODE='${m}';render()">${m==='fresh'?'fresh(学習なし)':'cumulative(学習込み)'}</span>`).join('')||'<span class="sub">まだデータがありません</span>';
  document.getElementById('metricChips').innerHTML =
    METRICS.map(([k,label])=>`<span class="chip ${k===METRIC?'on':''}" onclick="METRIC='${k}';render()">${label}</span>`).join('');
}

function render(){
  const rows=DATA.filter(r=>r.mode===MODE);
  renderChips();
  const a=agg(rows);
  renderCards(a);
  renderChart(a);
  renderTable(a);
  renderRaw(rows);
}

async function refresh(){
  try{
    const res=await fetch('/api/data');
    DATA=(await res.json()).runs;
    render();
  }catch(e){}
}
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/api/data'):
            rows = []
            if os.path.exists(BENCH_DB):
                db = sqlite3.connect(f'file:{BENCH_DB}?mode=ro', uri=True)
                db.row_factory = sqlite3.Row
                rows = [dict(r) for r in db.execute(
                    "SELECT * FROM bench_runs ORDER BY created")]
                db.close()
            body = json.dumps({'runs': rows}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description='Benchmark web dashboard')
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    print(f"Benchmark dashboard: http://localhost:{args.port}")
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
