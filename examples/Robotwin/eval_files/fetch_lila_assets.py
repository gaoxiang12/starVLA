"""Download official reproduction assets, checking publisher SHA-256 hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repository, destination, names, workers):
    destination.mkdir(parents=True, exist_ok=True)
    base = f"https://www.modelscope.cn/api/v1/models/{repository}/repo"
    session = requests.Session()
    session.trust_env = False
    response = session.get(base + "/files", params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    manifest = response.json()
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for item in manifest["Data"]["Files"]:
        if item["Path"] not in names:
            continue
        target = destination / item["Path"]
        if target.is_file() and sha256(target) == item["Sha256"]:
            print("Already verified", target, flush=True)
            continue
        size = item["Size"]
        part_size = 4 * 1024 * 1024
        parts = destination / (target.name + ".parts")
        parts.mkdir(exist_ok=True)

        def download(index):
            start, end = index * part_size, min((index + 1) * part_size, size) - 1
            path = parts / f"{index:05d}"
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(5):
                try:
                    with requests.Session() as client:
                        client.trust_env = False
                        r = client.get(base, params={"Revision": item["Revision"], "FilePath": item["Path"]},
                                       headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 45))
                        r.raise_for_status()
                        if len(r.content) != end - start + 1:
                            raise ValueError("Server did not honor requested range")
                        if r.status_code == 206 and r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError("Incorrect Content-Range")
                        path.write_bytes(r.content)
                        return path
                except (requests.RequestException, ValueError):
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 1)

        count = (size + part_size - 1) // part_size
        print("Downloading", repository, target.name, size, "bytes", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(download, range(count)))
        temporary = destination / (target.name + ".assembled")
        with temporary.open("wb") as output:
            for path in paths:
                output.write(path.read_bytes())
        if sha256(temporary) != item["Sha256"]:
            raise ValueError(f"Publisher SHA-256 mismatch: {target}")
        temporary.replace(target)
        for path in paths:
            path.unlink()
        parts.rmdir()
        print("Verified", target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/gaoxiang/ckpts"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    fetch("yangfan97/LiLa-WAM_RoboTwin2_0", args.root / "LiLa-WAM_RoboTwin2_0",
          {"config.yaml", "checkpoint.pt", "README.md"}, args.workers)
    fetch("facebook/dinov3-vitl16-pretrain-lvd1689m", args.root / "dinov3-vitl16-pretrain-lvd1689m",
          {"config.json", "model.safetensors", "preprocessor_config.json", "LICENSE.md"}, args.workers)


if __name__ == "__main__":
    main()
