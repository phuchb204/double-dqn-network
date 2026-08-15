import csv
import os
import time

from .dqn import DQNAgent
from .environment import RoutingEnv
from .evaluate import make_rl_select, run_policy, summarize

CSV_FIELDS = [
    "episode", "train_steps", "train_reward", "train_loss", "eps",
    "train_avg_delay_ms", "train_loss_rate", "train_throughput",
    "eval_reward", "eval_avg_delay_ms", "eval_loss_rate", "eval_throughput",
]


def train(env_cfg, agent_cfg, n_episodes, eval_every, eval_episodes,
          out_csv, out_model=None, seed=0, eval_seed_offset=100000, log_every=1):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    env = RoutingEnv(**env_cfg)
    agent = DQNAgent(env.state_dim, env.action_dim, agent_cfg)
    rl_select = make_rl_select(agent)

    file_mode = "w"
    if os.path.exists(out_csv):
        file_mode = "a"
    with open(out_csv, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        for ep in range(1, n_episodes + 1):
            t0 = time.time()
            obs, mask, done, _ = env.reset(seed=seed + ep)
            ep_rew = 0.0
            steps = 0
            losses = []
            while not done:
                action = agent.act(obs, mask)
                next_obs, next_mask, reward, done, info = env.step(action)
                agent.store(obs, action, reward, next_obs, done, next_mask)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                obs, mask = next_obs, next_mask
                steps += 1
                ep_rew += reward

            row = {
                "episode": ep,
                "train_steps": steps,
                "train_reward": round(ep_rew, 4),
                "train_loss": round(float(np_mean(losses)), 6) if losses else "",
                "eps": round(agent.epsilon, 4),
                "train_avg_delay_ms": round(info.get("avg_delay_ms", 0.0), 4),
                "train_loss_rate": round(info.get("packet_loss_rate", 0.0), 4),
                "train_throughput": round(info.get("throughput", 0.0), 4),
                "eval_reward": "",
                "eval_avg_delay_ms": "",
                "eval_loss_rate": "",
                "eval_throughput": "",
            }

            if ep % eval_every == 0:
                eval_rows = run_policy(env, rl_select, eval_episodes, seed=eval_seed_offset)
                summ = summarize(eval_rows)
                row["eval_reward"] = round(summ["episode_reward"], 4)
                row["eval_avg_delay_ms"] = round(summ["avg_delay_ms"], 4)
                row["eval_loss_rate"] = round(summ["packet_loss_rate"], 4)
                row["eval_throughput"] = round(summ["throughput"], 4)
                if out_model:
                    model_path = out_model if ep == n_episodes else f"{out_model}.ep{ep}"
                    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
                    agent.save(model_path)

            writer.writerow(row)
            f.flush()
            if ep % log_every == 0 or ep == n_episodes:
                print(f"[{tag_out(out_csv)}] ep {ep}/{n_episodes} "
                      f"reward={row['train_reward']} loss={row['train_loss']} "
                      f"eps={row['eps']} delay={row['train_avg_delay_ms']}ms "
                      f"loss_rate={row['train_loss_rate']} ({time.time() - t0:.1f}s)",
                      flush=True)
    return agent


def np_mean(values):
    import numpy as np
    return float(np.mean(values))


def tag_out(out_csv):
    return os.path.basename(out_csv).replace(".csv", "")
