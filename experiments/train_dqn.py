import argparse

from _common import add_common_args
from src.train import train

parser = argparse.ArgumentParser(description="Train DQN baseline")
add_common_args(parser)
args = parser.parse_args()

tag = args.tag or f"dqn_load{args.load}"
env_cfg = dict(n_nodes=args.n_nodes, seed=args.seed, topology=args.topology,
               buffer=args.buffer, mu=args.mu, load=args.load,
               n_flows=args.n_flows, total_rounds=args.total_rounds)
agent_cfg = dict(hidden=args.hidden, lr=args.lr, gamma=args.gamma, batch_size=args.batch_size,
                 buffer_capacity=args.buffer_capacity, eps_start=args.eps_start,
                 eps_end=args.eps_end, eps_decay_steps=args.eps_decay_steps,
                 target_update_freq=args.target_update_freq, device=args.device,
                 seed=args.run_seed, use_double=False)

out_csv = f"{args.out_dir}/csv/{tag}.csv"
out_model = f"{args.out_dir}/models/{tag}.pt"

agent = train(env_cfg, agent_cfg, n_episodes=args.episodes, eval_every=args.eval_every,
              eval_episodes=args.eval_episodes, out_csv=out_csv, out_model=out_model,
              seed=args.run_seed)
agent.save(out_model)
print(f"Saved model -> {out_model}")
print(f"Saved log  -> {out_csv}")
