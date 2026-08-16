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
    m = re.match(r"(dqn|ddqn)_load([0-9.]+)_s([0-9]+)", name)
    if m:
        return m.group(1), float(m.group(2)), int(m.group(3))
    m = re.match(r"(dqn|ddqn)_load([0-9.]+)", name)
    if m:
        return m.group(1), float(m.group(2)), 0
    return None


def load_training(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "dqn*.csv")) + glob.glob(os.path.join(csv_dir, "ddqn*.csv"))
    dfs = {}
    for f in files:
        tag = parse_tag(f)
        if tag is None:
            continue
        algo, load, seed = tag
        df = pd.read_csv(f)
        df["algo"] = algo
        df["seed"] = seed
        dfs.setdefault(load, []).append(df)
    return dfs


def load_baseline(csv_dir):
    path = os.path.join(csv_dir, "baseline.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _series_mean_std(dfs, column, window=20):
    merged = pd.concat([d[["episode", column]].dropna() for d in dfs], ignore_index=True)
    g = merged.groupby("episode")[column].agg(["mean", "std"]).reset_index()
    mean = g["mean"].rolling(window, min_periods=1).mean()
    std = g["std"].rolling(window, min_periods=1).mean()
    return g["episode"].values, mean.values, std.values


def plot_training_curves(training, out_dir):
    colors = {"dqn": "#1f77b4", "ddqn": "#ff7f0e"}
    for load, dfs in training.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        specs = [
            (axes[0, 0], "train_reward", "Reward/episode"),
            (axes[0, 1], "train_loss", "Loss"),
            (axes[1, 0], "train_avg_delay_ms", "Avg delay (ms)"),
            (axes[1, 1], "train_loss_rate", "Packet loss rate"),
        ]
        for algo in ["dqn", "ddqn"]:
            sub = [d for d in dfs if d["algo"].iloc[0] == algo]
            if not sub:
                continue
            c = colors[algo]
            for ax, col, title in specs:
                x, y, s = _series_mean_std(sub, col)
                ax.plot(x, y, label=algo, color=c)
                ax.fill_between(x, y - s, y + s, color=c, alpha=0.15)
                ax.set_title(title)
        for algo in ["dqn", "ddqn"]:
            sub = [d for d in dfs if d["algo"].iloc[0] == algo]
            if not sub:
                continue
            c = colors[algo]
            for df in sub:
                ev = df.dropna(subset=["eval_reward"])
                if len(ev):
                    axes[0, 0].plot(ev["episode"], ev["eval_reward"], "o", color=c, alpha=0.4)
        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[1, 0].legend()
        axes[1, 1].legend()
        for ax in axes.flat:
            ax.set_xlabel("episode")
        fig.suptitle(f"Training curves (load={load}, mean±std over seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"training_load{load}.png"), dpi=150)
        plt.close(fig)
    print(f"Saved training curves -> {out_dir}")


def final_eval_summary(training, baseline):
    rows = []
    for load, dfs in training.items():
        for algo in ["dqn", "ddqn"]:
            sub = [d for d in dfs if d["algo"].iloc[0] == algo]
            if not sub:
                continue
            for df in sub:
                ev = df.dropna(subset=["eval_reward"])
                if len(ev) == 0:
                    continue
                last = ev.tail(1).iloc[0]
                rows.append({"load": load, "algo": algo, "seed": df["seed"].iloc[0],
                             "reward": last["eval_reward"],
                             "delay_ms": last["eval_avg_delay_ms"],
                             "loss_rate": last["eval_loss_rate"],
                             "throughput": last["eval_throughput"]})
    if baseline is not None:
        for _, row in baseline.iterrows():
            for s in [0]:
                rows.append({"load": row["load"], "algo": "dijkstra", "seed": s,
                             "reward": row["episode_reward"], "delay_ms": row["avg_delay_ms"],
                             "loss_rate": row["packet_loss_rate"], "throughput": row["throughput"]})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    g = df.groupby(["load", "algo"])
    out = pd.DataFrame({
        "reward": g["reward"].mean(),
        "reward_std": g["reward"].std(),
        "delay_ms": g["delay_ms"].mean(),
        "delay_ms_std": g["delay_ms"].std(),
        "loss_rate": g["loss_rate"].mean(),
        "loss_rate_std": g["loss_rate"].std(),
        "throughput": g["throughput"].mean(),
        "throughput_std": g["throughput"].std(),
        "n_seeds": g.size(),
    }).reset_index()
    return out


def plot_comparison_bars(summary, out_dir):
    if len(summary) == 0:
        return
    metrics = [("reward", "reward_std"), ("delay_ms", "delay_ms_std"),
               ("loss_rate", "loss_rate_std"), ("throughput", "throughput_std")]
    for load in sorted(summary["load"].unique()):
        sub = summary[summary["load"] == load]
        algos = sorted(sub["algo"].unique())
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (metric, metric_std) in zip(axes, metrics):
            for i, algo in enumerate(algos):
                row = sub[sub["algo"] == algo].iloc[0]
                err = row.get(metric_std, 0.0) or 0.0
                ax.bar(i, row[metric], width=0.5, yerr=err, capsize=4, label=algo)
            ax.set_title(metric)
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(algos)
            ax.legend()
        fig.suptitle(f"Final comparison (load={load}, mean±std over seeds)")
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
