import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_env_cfg():
    return dict(n_nodes=10, seed=0, buffer=8, mu=3, load=0.7, n_flows=8,
                total_rounds=40, time_per_round=5.0)


def add_common_args(parser):
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="seed topology")
    parser.add_argument("--topology", type=str, default="hub",
                        help="hub (bottleneck tai node 0) | random")
    parser.add_argument("--buffer", type=int, default=8)
    parser.add_argument("--mu", type=int, default=3, help="service rate (packets/round)")
    parser.add_argument("--load", type=float, default=0.7, help="traffic intensity")
    parser.add_argument("--n-flows", type=int, default=8)
    parser.add_argument("--total-rounds", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=20000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="")
