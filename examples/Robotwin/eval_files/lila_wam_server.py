"""Serve an official LiLa-WAM checkpoint using StarVLA WebSockets."""
import argparse
import logging
from omegaconf import OmegaConf
import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.WM4A.LiLaWAM import LiLaWAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=5794)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    if cfg.framework.get('name') == 'GAWMOfficial':
        from examples.LiLaWAM.gawm_official import GAWMOfficialPolicy
        framework = GAWMOfficialPolicy(cfg)
    else:
        framework = LiLaWAM(cfg)
    WebsocketPolicyServer(framework, host="127.0.0.1", port=args.port,
                          metadata=dict(framework.metadata, policy_seed=args.seed)).serve_forever()


if __name__ == "__main__":
    main()
