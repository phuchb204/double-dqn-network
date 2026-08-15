import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_tag(path):
    name = os.path.basename(path).replace(".csv", "")
    m = re.match(r"(dqn|ddqn)_load([0-9.]+)", name)
    if not m:
        return None
    return m.group(1), float(m.group(2))


def load_training(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "dqn*.csv")) + glob.glob(os.path.join(csv_dir, "ddqn*.csv"))
    dfs = {}
    for f in files:
        tag = parse_tag(f)
        if tag is None:
            continue
        algo, load = tag
        df = pd.read_csv(f)
        df["algo"] = algo
        dfs.setdefault(load, []).append((algo, df))
    return dfs


def load_baseline(csv_dir):
    path = os.path.join(csv_dir, "baseline.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def plot_training_curves(training, out_dir):
    colors = {"dqn": "#1f77b4", "ddqn": "#ff7f0e"}
    for load, entries in training.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for algo, df in entries:
            c = colors[algo]
            axes[0, 0].plot(df["episode"], df["train_reward"].rolling(20, min_periods=1).mean(), label=f"{algo}", color=c)
            axes[0, 1].plot(df["episode"], df["train_loss"].rolling(20, min_periods=1).mean(), label=f"{algo}", color=c)
            axes[1, 0].plot(df["episode"], df["train_avg_delay_ms"].rolling(20, min_periods=1).mean(), label=f"{algo}", color=c)
            axes[1, 1].plot(df["episode"], df["train_loss_rate"].rolling(20, min_periods=1).mean(), label=f"{algo}", color=c)
            if "eval_reward" in df and df["eval_reward"].notna().any():
                ev = df.dropna(subset=["eval_reward"])
                axes[0, 0].plot(ev["episode"], ev["eval_reward"], "o", color=c, alpha=0.5, label=f"{algo} (eval)")
        axes[0, 0].set_title("Reward/episode"); axes[0, 0].legend()
        axes[0, 1].set_title("Loss"); axes[0, 1].legend()
        axes[1, 0].set_title("Avg delay (ms)"); axes[1, 0].legend()
        axes[1, 1].set_title("Packet loss rate"); axes[1, 1].legend()
        for ax in axes.flat:
            ax.set_xlabel("episode")
        fig.suptitle(f"Training curves (load={load})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"training_load{load}.png"), dpi=150)
        plt.close(fig)
    print(f"Saved training curves -> {out_dir}")


def final_eval_summary(training, baseline):
    rows = []
    for load, entries in training.items():
        for algo, df in entries:
            ev = df.dropna(subset=["eval_reward"])
            if len(ev) == 0:
                continue
            last = ev.tail(1).iloc[0]
            rows.append({"load": load, "algo": algo,
                         "reward": last["eval_reward"],
                         "delay_ms": last["eval_avg_delay_ms"],
                         "loss_rate": last["eval_loss_rate"],
                         "throughput": last["eval_throughput"]})
    if baseline is not None:
        for _, row in baseline.iterrows():
            rows.append({"load": row["load"], "algo": "dijkstra",
                         "reward": row["episode_reward"], "delay_ms": row["avg_delay_ms"],
                         "loss_rate": row["packet_loss_rate"], "throughput": row["throughput"]})
    return pd.DataFrame(rows)


def plot_comparison_bars(summary, out_dir):
    if len(summary) == 0:
        return
    metrics = ["reward", "delay_ms", "loss_rate", "throughput"]
    widths = 0.25
    for load in sorted(summary["load"].unique()):
        sub = summary[summary["load"] == load]
        algos = sorted(sub["algo"].unique())
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, metric in zip(axes, metrics):
            for i, algo in enumerate(algos):
                row = sub[sub["algo"] == algo].iloc[0]
                ax.bar(i, row[metric], width=widths, label=algo)
            ax.set_title(metric)
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(algos)
            ax.legend()
        fig.suptitle(f"Final comparison (load={load})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"comparison_load{load}.png"), dpi=150)
        plt.close(fig)
    print(f"Saved comparison bars -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from results/csv")
    parser.add_argument("--csv-dir", default="results/csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    training = load_training(args.csv_dir)
    baseline = load_baseline(args.csv_dir)
    plot_training_curves(training, args.out_dir)
    summary = final_eval_summary(training, baseline)
    plot_comparison_bars(summary, args.out_dir)
    summary.to_csv(os.path.join(args.csv_dir, "summary.csv"), index=False)
    print(f"Saved summary -> {args.csv_dir}/summary.csv")


if __name__ == "__main__":
    main()
