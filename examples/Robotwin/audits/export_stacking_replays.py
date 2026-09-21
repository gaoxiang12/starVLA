"""Export completed stacking recordings and a local video viewing page."""
import html
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent / 'stacking_replays_20260909'
NAMES = {'stack_blocks_two': '两个积木堆叠', 'stack_bowls_two': '两个碗堆叠',
         'stack_blocks_three': '三个积木堆叠', 'stack_bowls_three': '三个碗堆叠'}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    sections, verified = [], []
    for job in manifest['jobs']:
        task = job['task']
        source = ROOT / task
        meta = json.loads((source / 'metadata.json').read_text())
        assert meta['complete'] and meta['seed'] == job['seed']
        assert meta['frames'] == meta['actions'] + 1
        output = ROOT / f'{task}_seed{meta["seed"]}.mp4'
        outcome = 'SUCCESS' if meta['success'] else 'FAILURE'
        vf = ['pad=iw:ih+68:0:44:color=black']
        for label, x, y, size in [
            (f'{task} | 40k | seed {meta["seed"]} | {outcome}', 10, 4, 17),
            ('Head camera', 110, 25, 14), ('Left wrist', 435, 25, 14),
            ('Right wrist', 745, 25, 14),
            (f'Action %{{n}} / {meta["actions"]} | 20 fps playback; not physical real time', 10, 'h-20', 14),
        ]:
            vf.append(f"drawtext=fontfile={FONT}:text='{label}':x={x}:y={y}:fontsize={size}:fontcolor=white")
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source / 'recording.mp4'),
                        '-vf', ','.join(vf), '-an', '-c:v', 'libx264', '-threads', '2',
                        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', '-y', str(output)], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,nb_frames,duration,pix_fmt',
            '-of', 'json', str(output)], text=True))['streams'][0]
        assert probe['codec_name'] == 'h264' and int(probe['nb_frames']) == meta['frames']
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
        # Contact sheet of the head camera for visual inspection of the full sequence.
        indices = sorted({round((meta['frames'] - 1) * k / 5) for k in range(6)})
        select = '+'.join(f'eq(n\\,{i})' for i in indices)
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source / 'recording.mp4'),
                        '-vf', f'crop=320:240:0:0,select={select},tile=3x2', '-frames:v', '1',
                        '-y', str(ROOT / f'{task}_contact.png')], check=True)
        success = '成功' if meta['success'] else '失败'
        original = '成功' if job['original_success'] else '失败'
        sections.append(f'<section><h2>{NAMES[task]}：本次重放{success}</h2>'
            f'<p>种子 {meta["seed"]}；原评测结果：{original}；完整回合 {meta["actions"]} 个动作步，'
            f'视频 {float(probe["duration"]):.1f} 秒。</p>'
            f'<video controls preload="metadata" src="{html.escape(output.name)}"></video>'
            f'<p><a href="{html.escape(output.name)}" download>下载视频</a></p></section>')
        verified.append(dict(task=task, metadata=meta, video=output.name, probe=probe,
                             original_success=job['original_success']))
    (ROOT / 'verified.json').write_text(json.dumps(verified, indent=2, ensure_ascii=False) + '\n')
    (ROOT / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>积木与碗：GAWM 回放对比</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.65 sans-serif;background:#151719;color:#eee}video{width:100%;background:#000}a{color:#82bdff}section{margin:36px 0}</style>
<h1>积木与碗：GAWM 40k 模型回放</h1>
<p>使用原版 49 任务 40k checkpoint，重新运行四个指定场景。原评测未保存视频，因此这些是本次重放记录。
案例按原评测的成功或失败结果选取，不能用于估计成功率。同一种子用于不同任务，也不代表物体布局完全相同。</p>
<p>每条展示完整回合。三个视角依次为头部、左腕、右腕；每个动作步一帧，20 fps 播放，不代表物理实时速度。</p>
''' + '\n'.join(sections) + '</html>')
    print(json.dumps([dict(task=v['task'], success=v['metadata']['success'],
                          duration=v['probe']['duration'], video=v['video']) for v in verified], indent=2))


if __name__ == '__main__':
    main()
