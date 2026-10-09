"""DINOv3 pretrained assets: ModelScope download with publisher hash checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from filelock import FileLock
import requests

MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"
DEFAULT_ENCODER_SPEC = "vitl16"
DINO_REPOSITORIES = {
    spec: f"facebook/dinov3-{spec}-pretrain-lvd1689m"
    for spec in ("vits16", "vits16plus", "vitb16", "vitl16")
}
DINO_FILES = {"config.json", "model.safetensors", "preprocessor_config.json", "LICENSE.md"}
DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[2] / "playground/Pretrained"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repository, destination, names, workers):
    destination = Path(destination)
    if workers < 1:
        raise ValueError("workers must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    base = f"{MODELSCOPE_ENDPOINT}/api/v1/models/{repository}/repo"
    session = requests.Session()
    session.trust_env = False
    response = session.get(base + "/files", params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    manifest = response.json()
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    missing = set(names) - {item["Path"] for item in manifest["Data"]["Files"]}
    if missing:
        raise ValueError(f"ModelScope repository {repository} is missing files: {sorted(missing)}")
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



def download_dinov3(encoder_spec=DEFAULT_ENCODER_SPEC, destination=None, workers=4):
    """Fetch a verified ModelScope snapshot; never fall back to another hub."""
    if encoder_spec not in DINO_REPOSITORIES:
        raise ValueError(f"Unsupported DINOv3 encoder spec: {encoder_spec}")
    repository = DINO_REPOSITORIES[encoder_spec]
    target = Path(destination).expanduser() if destination is not None else DEFAULT_ASSET_ROOT / repository.split("/")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        fetch(repository, target, DINO_FILES, workers)
    return str(target.resolve())


def resolve_dinov3_path(path):
    """Preserve local paths; resolve explicit facebook/dinov3 IDs via ModelScope."""
    local = Path(path).expanduser()
    if local.is_dir():
        return str(local.resolve())
    specs = {repository: spec for spec, repository in DINO_REPOSITORIES.items()}
    if str(path) not in specs:
        raise FileNotFoundError(f"DINOv3 local directory does not exist: {path}")
    cached = DEFAULT_ASSET_ROOT / str(path).split("/")[-1]
    if all((cached / name).is_file() for name in DINO_FILES):
        return str(cached.resolve())
    if any(os.environ.get(key, "").upper() in {"1", "TRUE", "YES", "ON"}
           for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise FileNotFoundError(f"DINOv3 is not cached at {cached}; download it before offline training")
    return download_dinov3(specs[str(path)])
