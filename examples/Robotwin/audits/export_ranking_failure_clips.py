"""Add view/step labels and export browser-compatible diagnostic MP4s."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent / 'ranking_failure_20260907/videos'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

for task, label in [('blocks_ranking_rgb', 'RGB ranking'), ('blocks_ranking_size', 'Size ranking')]:
    source = ROOT / task
    meta = json.loads((source / 'metadata.json').read_text())
    assert meta['actions'] == 320 and meta['frames'] == 321
    assert not meta['success_during_prefix']
    output = ROOT / f'{task}_seed100000_first320.mp4'
    filters = ['pad=iw:ih+80:0:40:color=black']
    for text, x, y, size in [
        (f'{label} | 40k model | seed 100000 | failed-rollout excerpt', 10, 5, 18),
        ('Head camera', 105, 24, 13), ('Left wrist', 430, 24, 13), ('Right wrist', 745, 24, 13),
        ('Action %{n} / 320    |    10 fps playback; one frame per action', 10, 'h-27', 16),
    ]:
        filters.append(f"drawtext=fontfile={FONT}:text='{text}':x={x}:y={y}:fontsize={size}:fontcolor=white")
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(source / 'recording.mp4'),
                    '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', str(output)], check=True)
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=codec_name,width,height,nb_frames,duration,pix_fmt',
                        '-of', 'json', str(output)], text=True))
    stream = result['streams'][0]
    assert int(stream['nb_frames']) == 321 and stream['codec_name'] == 'h264'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    (ROOT / f'{task}_video_verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output, output.stat().st_size, stream, flush=True)

(ROOT / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>排序任务失败回放</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:17px/1.6 sans-serif;background:#151719;color:#eee}video{width:100%;background:#000}a{color:#82bdff}section{margin:32px 0}</style>
<h1>40k 模型：排序任务失败回放</h1>
<p>同一失败种子 100000，分别重放前 320 个动作步。每个动作步记录一帧，以 10 fps 播放；每条约 32 秒。三视角依次为头部、左腕、右腕。原始正式评测中这两个种子的完整回合均失败；这里展示首抓及后续动作片段。</p>
<section><h2>RGB 排序</h2><p>约 7–9 秒：夹爪在红、蓝块之间闭合；随后空手移动。</p>
<video controls preload="metadata" src="blocks_ranking_rgb_seed100000_first320.mp4"></video>
<a href="blocks_ranking_rgb_seed100000_first320.mp4" download>下载 RGB 排序视频</a></section>
<section><h2>尺寸排序</h2><p>约 5–9 秒：右臂接近中号块，抓取未有效提起积木；后续仍有动作。</p>
<video controls preload="metadata" src="blocks_ranking_size_seed100000_first320.mp4"></video>
<a href="blocks_ranking_size_seed100000_first320.mp4" download>下载尺寸排序视频</a></section></html>''')
