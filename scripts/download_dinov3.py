"""Download pretrained DINOv3 from ModelScope into playground/Pretrained."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.model.dinov3_assets import DEFAULT_ENCODER_SPEC, DINO_REPOSITORIES, download_dinov3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-spec', choices=DINO_REPOSITORIES, default=DEFAULT_ENCODER_SPEC)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(download_dinov3(args.encoder_spec, args.output_dir, args.workers))


if __name__ == '__main__':
    main()
