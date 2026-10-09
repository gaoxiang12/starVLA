"""Download official reproduction assets, checking publisher SHA-256 hashes."""
import argparse
from pathlib import Path
import sys

# Keep direct script invocation working outside an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from starVLA.model.dinov3_assets import download_dinov3, fetch, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("playground/Pretrained"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    fetch("yangfan97/LiLa-WAM_RoboTwin2_0", args.root / "LiLa-WAM_RoboTwin2_0",
          {"config.yaml", "checkpoint.pt", "README.md"}, args.workers)
    download_dinov3("vitl16", args.root / "dinov3-vitl16-pretrain-lvd1689m", args.workers)


if __name__ == "__main__":
    main()
