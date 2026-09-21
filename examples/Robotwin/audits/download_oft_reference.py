"""Download pinned public OFT/Qwen assets once into the shared checkpoint store."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    os.environ.pop(key, None)

from huggingface_hub import HfApi, snapshot_download
import psutil

from examples.Robotwin.audits.run_grasp_lift_development import ROOT, digest, save


def main():
    out = ROOT/'playground/Checkpoints/oft_grasp_precision_reference_20260909'
    out.mkdir(exist_ok=True)
    status = out/'download_status.json'
    assert not status.exists()
    assets = [
        ('StarVLA/Qwen3-VL-OFT-RoboTwin2-All', '9fae49581755f57944ec57f6787d214c0c0143f5',
         '/data/gaoxiang/ckpts/Qwen3-VL-OFT-RoboTwin2-All',
         ['README.md','config.yaml','dataset_statistics.json','summary.jsonl','checkpoints/steps_140000_pytorch_model.pt']),
        ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
         '/data/gaoxiang/ckpts/Qwen3-VL-4B-Instruct',
         ['*.json','*.txt','*.safetensors']),
    ]
    report = dict(state='downloading', pid=os.getpid(), birth=psutil.Process().create_time(),
                  endpoint=os.environ['HF_ENDPOINT'], completed=[])
    save(status, report)
    try:
        for repo, revision, directory, patterns in assets:
            report.update(current_repo=repo, revision=revision, directory=directory)
            save(status, report)
            info = HfApi(endpoint=os.environ['HF_ENDPOINT']).model_info(repo, revision=revision, files_metadata=True)
            assert info.sha == revision
            snapshot_download(repo_id=repo, revision=revision, local_dir=directory,
                              allow_patterns=patterns, max_workers=3, endpoint=os.environ['HF_ENDPOINT'])
            files = []
            for entry in info.siblings:
                path = Path(directory)/entry.rfilename
                if not path.is_file():
                    continue
                sha = digest(path)
                assert path.stat().st_size == entry.size
                if entry.lfs is not None:
                    assert sha == entry.lfs.sha256, f'Weight SHA256 mismatch: {path}'
                files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha))
            report['completed'].append(dict(repo=repo, revision=revision, directory=directory, files=files))
            save(status, report)
        report['state'] = 'complete'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        report['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save(status, report)


if __name__ == '__main__':
    main()
