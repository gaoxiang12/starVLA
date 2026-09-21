"""Export the first completed paired grasp scenes, without selecting by outcome."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def frame_sim_times(result, actions):
    """A physics-budget exception can interrupt the last action before filming."""
    assert result['state'] == 'complete' and actions
    times = [float(a['sim_s']) for a in actions]
    if result['frames'] == len(actions)+1:
        times.append(float(result['result']['elapsed_sim_s']))
    else:
        assert result['frames'] == len(actions) and result['termination']=='simulation_time_budget'
    assert len(times)==result['frames'] and times[0]==0 and np.all(np.diff(times)>0)
    return times


def video_record(case, output):
    case = Path(case)
    result = json.loads((case/'result.json').read_text())
    actions = json.loads((case/'action_trace.json').read_text())
    times = frame_sim_times(result,actions)
    source = case/'rollout.mp4'
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if output.exists():
        assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    else:
        shutil.copyfile(source,output)
    with av.open(str(output)) as video:
        stream = video.streams.video[0]
        assert float(stream.average_rate) == 20 and (stream.width,stream.height)==(960,240)
        decoded = sum(1 for _ in video.decode(video=0))
    assert decoded == result['frames']
    # frame 0 is initial observation; following frames are action-end states.
    assert len(times) == decoded
    return dict(file=output.name,source=str(source.resolve()),sha256=digest,frames=decoded,
        fps=20,frame_sim_times=times,checkpoint=result['policy'],result=result['result'],
        physical_end_time_s=result['result']['elapsed_sim_s'],
        final_action_interrupted_before_video_frame=result['frames']==len(actions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=4)
    args = parser.parse_args()
    assert 1 <= args.limit <= 20
    args.output.mkdir(parents=True,exist_ok=True)
    statuses = [json.loads((p/'status.json').read_text()) for p in (args.reference,args.candidate)]
    indexes = [{r['seed']:r for r in s['results']} for s in statuses]
    selected = sorted(set(indexes[0]) & set(indexes[1]))[:args.limit]
    assert selected
    records,sections = [],[]
    labels = ['原 RGB 模型（v2）','原动作头＋抓取专项训练 1000 步']
    for seed in selected:
        pair = [index[seed] for index in indexes]
        cases = [json.loads((Path(r['source_case'])/'result.json').read_text()) for r in pair]
        assert cases[0]['initial_scene_audit']['initial_rgb_sha256'] == cases[1]['initial_scene_audit']['initial_rgb_sha256']
        for key in ('initial_block_positions_m','initial_block_quaternions','block_half_extents'):
            np.testing.assert_array_equal(cases[0][key],cases[1][key])
        videos = [video_record(r['source_case'],args.output/f'seed_{seed}_{tag}.mp4')
                  for r,tag in zip(pair,['reference','joint1000'])]
        cells=[]
        for label,r,video in zip(labels,pair,videos):
            outcome='首抓成功' if r['scored_result']['first_attempt_success'] else '第二次抓取成功' if r['scored_result']['success'] else '失败'
            cells.append(f'<div><h3>{html.escape(label)} · {outcome}</h3><video id="v{seed}_{len(cells)}" controls preload="metadata" src="{video["file"]}"></video><p>当前记录帧：<output id="frame{seed}_{len(cells)}">0.00 s</output></p><a download href="{video["file"]}">下载视频</a></div>')
        table=[]
        for title,key in [('开始闭合前','preclosure'),('drive 降至 80%','0.8'),('drive 降至 50%','0.5'),('drive 降至 20%','0.2'),('drive 降至 5%','0.05')]:
            values=[]
            for r in pair:
                event=r['preclosure'] if key=='preclosure' else r['drive_threshold_events'].get(key)
                values.append(f'{event["horizontal_offset_mm"]:.2f} mm' if event else '未发生')
            table.append(f'<tr><td>{title}</td><td>{values[0]}</td><td>{values[1]}</td></tr>')
        maximum=max(v['frame_sim_times'][-1] for v in videos)
        sections.append(f'<section><h2>场景 {seed} · 积木边长 {pair[0]["cube_edge_mm"][0]:.1f} mm</h2><div class="pair">'+''.join(cells)+f'</div><p>查看同一仿真时刻：<output id="time{seed}">0.00 s</output></p><input aria-label="仿真时间" type="range" min="0" max="{maximum}" step="0.02" value="0" oninput="seekPair({seed},Number(this.value))"><table><thead><tr><th>闭合阶段</th><th>原 RGB 模型</th><th>专项训练对照</th></tr></thead><tbody>'+''.join(table)+'</tbody></table></section>')
        records.append(dict(seed=seed,videos=videos,precision=pair))
    manifest=dict(state='partial_development_review',selected_seeds=selected,
        selection='First completed shared seeds in protocol order; no selection by success or error.',
        reference_completed=len(indexes[0]),candidate_completed=len(indexes[1]),planned=20,records=records,
        note='This compares training data/schedule with the original joint head. The Cartesian model is still training. Ordinary video playback is 20 recorded frames/s, not physical time; slider aligns recorded action-end observations by simulation time.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    data=json.dumps({str(r['seed']):[v['frame_sim_times'] for v in r['videos']] for r in records})
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>抓取精度：同场景回放对照</title>
<style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif;background:#141719;color:#eee}a{color:#86c1ff}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}video,input{width:100%}section{border-top:1px solid #596169;margin-top:30px;padding-top:15px}table{border-collapse:collapse;margin:20px 0}td,th{padding:8px 18px;border-bottom:1px solid #596169;text-align:left}h3{font-size:17px}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<h1>抓取精度：同场景回放对照</h1><p>当前展示最先完成的配对场景，未按成功或失败挑选。全部 20 场景尚未完成，不能用此页估计整体提升。右侧仍是原关节动作头；笛卡尔修正版正在训练。</p><p>三个视角依次为头部、左腕、右腕。普通播放为每个记录动作一帧、20 fps，不代表物理实时速度。时间滑条暂停视频，并选取不晚于该仿真时刻的最近记录帧；它不生成中间物理帧。</p><p>表格是实际 TCP 到当时红块中心的水平偏移，不是独立标注的最优抓取姿态误差。drive 为中间夹爪驱动目标，实际手指开度可能因接触而不同；按阶段比较，不能只用开始闭合时的数值代表最终对齐。</p>
'''+''.join(sections)+'''<script>const times='''+data+''';function seekPair(seed,time){document.getElementById('time'+seed).textContent=time.toFixed(2)+' s';times[seed].forEach((frames,j)=>{let i=0;while(i+1<frames.length&&frames[i+1]<=time)i++;const v=document.getElementById('v'+seed+'_'+j);v.pause();v.currentTime=i/20;});}Object.entries(times).forEach(([seed,pair])=>pair.forEach((frames,j)=>{const v=document.getElementById('v'+seed+'_'+j);v.addEventListener('timeupdate',()=>{const i=Math.min(frames.length-1,Math.floor(v.currentTime*20+.001));document.getElementById('frame'+seed+'_'+j).textContent=frames[i].toFixed(2)+' s'+(i===frames.length-1?'（最后记录帧）':'');});}));</script></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps(dict(selected=selected,page=str(args.output/'index.html'))),flush=True)


if __name__ == '__main__':
    main()
