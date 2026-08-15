import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import add_common_args
from src.evaluate import run_baseline

parser = argparse.ArgumentParser(description="Baseline Dijkstra shortest-path routing")
add_common_args(parser)
parser.add_argument("--loads", type=str, default="0.3,0.5,0.7,0.9,1.1")
parser.add_argument("--episodes", type=int, default=50)
args = parser.parse_args()

loads = [float(x) for x in args.loads.split(",")]
out_csv = f"{args.out_dir}/csv/baseline.csv"
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["load", "episode_reward", "avg_delay_ms", "packet_loss_rate",
                     "throughput", "arrived"])
    for load in loads:
        env_cfg = dict(n_nodes=args.n_nodes, seed=args.seed, buffer=args.buffer, mu=args.mu,
                       load=load, n_flows=args.n_flows, total_rounds=args.total_rounds)
        summ, _rows = run_baseline(env_cfg, n_episodes=args.episodes)
        writer.writerow([load, round(summ["episode_reward"], 4),
                         round(summ["avg_delay_ms"], 4),
                         round(summ["packet_loss_rate"], 4),
                         round(summ["throughput"], 4),
                         round(summ["arrived"], 4)])
        print(f"load={load}: reward={summ['episode_reward']:.3f} "
              f"delay={summ['avg_delay_ms']:.2f}ms "
              f"loss={summ['packet_loss_rate']:.3f} "
              f"throughput={summ['throughput']:.3f}", flush=True)

print(f"Saved -> {out_csv}")
