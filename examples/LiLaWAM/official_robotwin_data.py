"""Fetch and audit the author's RoboTwin training data (not the policy weights)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import time
import zipfile

import requests

REPOSITORY = 'yangfan97/LiLa-WAM_RoboTwin2.0_50_task'
API = f'https://www.modelscope.cn/api/v1/datasets/{REPOSITORY}/repo'
OUTLIERS = Path('/data/gaoxiang/Code/LiLa-WAM/utils/outlier_files 500-all.txt')


def removed_episodes(report=OUTLIERS):
    removed = {}
    for line in Path(report).read_text().splitlines():
        match = re.search(r'^\[Action\].*/([^/]+)/(demo_clean|demo_randomized)/data/episode_?(\d+)\.hdf5', line)
        if match:
            task, split, index = match.groups()
            removed.setdefault((task, split), set()).add(int(index))
    return removed


def expected_episode_count(report=OUTLIERS):
    return 27500 - sum(len(ids) for ids in removed_episodes(report).values())


def validate_episode_ids(task, names, report=OUTLIERS):
    removed = removed_episodes(report)
    missing_report = {}
    for split, count in [('demo_clean', 50), ('demo_randomized', 500)]:
        actual = []
        for name in names[split]:
            match = re.fullmatch(r'episode_?(\d+)\.hdf5', name)
            if not match:
                raise ValueError(f'Invalid episode name: {name}')
            actual.append(int(match[1]))
        excluded = removed.get((task, split), set())
        expected = set(range(count)) - excluded
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'{task}/{split}: unexpected published counts/IDs; '
                             f'missing={sorted(expected-set(actual))}, extra={sorted(set(actual)-expected)}')
        missing_report[split] = sorted(excluded)
    return missing_report


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(buf)
    return h.hexdigest()


def get_manifest(root):
    path = root / 'source_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    with requests.Session() as s:
        s.trust_env = False
        r = s.get(API + '/tree', params={'Revision': 'master', 'Root': '', 'Recursive': 'false'}, timeout=(15, 45))
        r.raise_for_status()
        value = r.json()
    files = [x for x in value['Data']['Files'] if x['Path'].endswith('.zip')]
    if len(files) != 50:
        raise ValueError(f'Expected 50 task archives, got {len(files)}')
    write_json(path, value)
    return value


def download(item, root, workers):
    archive = root / 'archives' / item['Name']
    marker = archive.with_suffix('.verified.json')
    if marker.exists() and archive.exists():
        prior = json.loads(marker.read_text())
        if prior['sha256'] == item['Sha256'] and archive.stat().st_size == item['Size']:
            return archive
    parts = archive.with_suffix('.parts')
    parts.mkdir(parents=True, exist_ok=True)
    size, block = item['Size'], 8 * 1024**2
    count = (size + block - 1) // block

    def part(i):
        begin, end = i * block, min(size, (i + 1) * block) - 1
        path = parts / f'{i:06d}'
        if path.exists() and path.stat().st_size == end - begin + 1:
            return path
        for attempt in range(8):
            try:
                with requests.Session() as s:
                    s.trust_env = False
                    r = s.get(API, params={'Revision': item['Revision'], 'FilePath': item['Path']},
                              headers={'Range': f'bytes={begin}-{end}'}, timeout=(15, 45))
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {begin}-{end}/{size}':
                        raise ValueError('Server did not honor range')
                    if len(r.content) != end - begin + 1:
                        raise ValueError('Incomplete range')
                    temp = path.with_suffix('.tmp')
                    temp.write_bytes(r.content)
                    temp.replace(path)
                    return path
            except (requests.RequestException, ValueError):
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 30))

    print(f'Downloading {item["Name"]}: {size / 1e9:.2f} GB', flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(part, i) for i in range(count)]
        for done, f in enumerate(as_completed(futures), 1):
            f.result()
            if done % 32 == 0 or done == count:
                write_json(root / 'download_status.json', {'task': item['Name'], 'parts_done': done,
                           'parts_total': count, 'elapsed_seconds': time.time() - started, 'updated': time.time()})
                print(f'{item["Name"]}: {done}/{count} parts', flush=True)
    temp = archive.with_suffix('.assembled')
    h = hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(count):
            data = (parts / f'{i:06d}').read_bytes()
            h.update(data)
            out.write(data)
    if h.hexdigest() != item['Sha256']:
        raise ValueError(f'SHA256 mismatch: {archive}')
    temp.replace(archive)
    write_json(marker, {'sha256': h.hexdigest(), 'size': size, 'repository': REPOSITORY, 'revision': item['Revision']})
    for i in range(count):
        (parts / f'{i:06d}').unlink()
    parts.rmdir()
    return archive


def extract(archive, task, root):
    """Extract only HDF5 episodes, mapping archive prefixes to a fixed layout."""
    marker = root / 'audits' / f'{task}.extraction.json'
    if marker.exists():
        return json.loads(marker.read_text())
    entries = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if '..' in p.parts or p.is_absolute():
                raise ValueError(f'Unsafe archive member: {p}')
            if p.suffix != '.hdf5':
                continue
            splits = [x for x in p.parts if x in ('demo_clean', 'demo_randomized')]
            if len(splits) != 1:
                raise ValueError(f'Unrecognized episode layout: {p}')
            target = root / 'data' / task / splits[0] / 'data' / p.name
            entries.append((info, target, splits[0]))
        if len({str(x[1]) for x in entries}) != len(entries):
            raise ValueError('Duplicate archive destinations')
        counts = {s: sum(x[2] == s for x in entries) for s in ('demo_clean', 'demo_randomized')}
        exclusions = validate_episode_ids(task, {s:[x[1].name for x in entries if x[2] == s] for s in counts})
        for info, target, split in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            temp = target.with_suffix('.partial')
            with z.open(info) as src, temp.open('wb') as out:
                for buf in iter(lambda: src.read(4 * 1024**2), b''):
                    out.write(buf)
            temp.replace(target)
    result = {'task': task, 'counts': counts, 'archive': str(archive), 'episodes': len(entries),
              'author_removed_episode_ids':exclusions, 'outlier_report_sha256':digest(OUTLIERS)}
    write_json(marker, result)
    return result


def audit_task(root, task):
    import cv2
    import h5py
    import numpy as np
    cv2.setNumThreads(1)
    records, hashes = [], {}
    for path in sorted((root / 'data' / task).glob('*/data/*.hdf5')):
        with h5py.File(path, 'r') as f:
            a = f['joint_action/vector'][:]
            state = np.concatenate([f['endpose/left_endpose'][:], f['endpose/left_gripper'][:].reshape(-1, 1),
                                    f['endpose/right_endpose'][:], f['endpose/right_gripper'][:].reshape(-1, 1)], axis=1)
            if len(a) < 2 or a.shape != (len(a), 14) or state.shape != (len(a), 16):
                raise ValueError(f'Shape/length mismatch: {path}')
            if not np.isfinite(a).all() or not np.isfinite(state).all():
                raise ValueError(f'Nonfinite state/action: {path}')
            rgb = f['observation/head_camera/rgb']
            if len(rgb) != len(a):
                raise ValueError(f'Image length mismatch: {path}')
            # Every image is decoded: a corrupted middle frame must not be silently skipped.
            for i, buf in enumerate(rgb):
                im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if im is None or im.ndim != 3 or im.shape[-1] != 3 or min(im.shape[:2]) < 16:
                    raise ValueError(f'Invalid head image: {path}, frame={i}')
            fingerprint = hashlib.sha256(a.tobytes() + state.tobytes()).hexdigest()
            if fingerprint in hashes:
                raise ValueError(f'Duplicate trajectory: {path}, {hashes[fingerprint]}')
            hashes[fingerprint] = str(path)
            records.append({'path': str(path.relative_to(root)), 'frames': len(a), 'state_action_sha256': fingerprint})
    exclusions = validate_episode_ids(task, {s:[Path(r['path']).name for r in records
        if s in Path(r['path']).parts] for s in ('demo_clean','demo_randomized')})
    result = {'task': task, 'episodes': len(records), 'frames': sum(r['frames'] for r in records),
              'head_images_checked': 'all frames', 'records': records, 'status': 'passed',
              'author_removed_episode_ids':exclusions, 'outlier_report_sha256':digest(OUTLIERS)}
    write_json(root / 'audits' / f'{task}.json', result)
    print(f'Audit passed: {task}, {result["frames"]} frames', flush=True)
    return result


def local_inventory(output):
    import pyarrow.parquet as pq
    rows = []
    for split in ('Clean', 'Randomized'):
        for folder in sorted((Path('/data/gaoxiang/RoboTwin') / split).iterdir()):
            info_path = folder / 'meta/info.json'
            info = json.loads(info_path.read_text())
            files = sorted(folder.glob('data/*/*.parquet'))
            schema = pq.read_schema(files[0])
            rows.append({'task': folder.name, 'split': split, 'episodes_metadata': info['total_episodes'],
                         'episode_files': len(files), 'frames_metadata': info['total_frames'],
                         'state_shape': info['features']['observation.state']['shape'],
                         'columns': schema.names, 'info_sha256': digest(info_path)})
    value = {'datasets': rows, 'episodes': sum(x['episode_files'] for x in rows),
             'frames_metadata': sum(x['frames_metadata'] for x in rows),
             'official_training_compatible': False,
             'reason': 'Converted data has 14D joint state, missing recorded 16D endpose state.'}
    write_json(output, value)
    print(json.dumps({k:v for k,v in value.items() if k != 'datasets'}, indent=2))


def prepare(root, workers=12):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = get_manifest(root)
    files = sorted([x for x in manifest['Data']['Files'] if x['Path'].endswith('.zip')], key=lambda x:x['Size'])
    # Audit completed downloads concurrently with fetching the next task.
    results = []
    with ThreadPoolExecutor(max_workers=2) as auditors:
        pending = []
        for item in files:
            task = Path(item['Name']).stem
            audit_path = root / 'audits' / f'{task}.json'
            if audit_path.exists():
                result = json.loads(audit_path.read_text())
                if result.get('status') == 'passed':
                    validate_episode_ids(task, {s:[Path(r['path']).name for r in result['records']
                        if s in Path(r['path']).parts] for s in ('demo_clean','demo_randomized')})
                    results.append(result)
                    continue
            archive = download(item, root, workers)
            extract(archive, task, root)
            pending.append(auditors.submit(audit_task, root, task))
            # Surface audit failures promptly, before fetching all remaining data.
            for future in pending:
                if future.done():
                    future.result()
        results.extend(f.result() for f in pending)
    result = {'status': 'passed', 'tasks': len(results), 'episodes': sum(x['episodes'] for x in results),
              'frames': sum(x['frames'] for x in results), 'task_summaries': [{k:v for k,v in x.items() if k != 'records'} for x in results]}
    result.update(expected_episodes=expected_episode_count(), collected_episodes=27500,
                  author_removed_episodes=27500-expected_episode_count(), outlier_report_sha256=digest(OUTLIERS))
    if result['tasks'] != 50 or result['episodes'] != expected_episode_count():
        raise ValueError(f'Incomplete dataset: {result}')
    write_json(root / 'dataset_audit.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/gaoxiang/LiLaWAM_RoboTwin_Official'))
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--inventory', type=Path)
    args = p.parse_args()
    if args.inventory:
        local_inventory(args.inventory)
    else:
        prepare(args.root, args.workers)
