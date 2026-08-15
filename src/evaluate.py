import numpy as np

from .environment import RoutingEnv


def dijkstra_next_hops(adj_matrix):
    n = adj_matrix.shape[0]
    table = {}
    for s in range(n):
        dist = np.full(n, np.inf)
        nxt = np.full(n, -1, dtype=int)
        dist[s] = 0.0
        visited = np.zeros(n, dtype=bool)
        for _ in range(n):
            u = -1
            best = np.inf
            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u == -1:
                break
            visited[u] = True
            for v in range(n):
                if adj_matrix[u, v] > 0 and not visited[v]:
                    cand = dist[u] + 1.0
                    if cand < dist[v]:
                        dist[v] = cand
                        nxt[v] = u
        for d in range(n):
            if d == s:
                continue
            hop = d
            while nxt[hop] != -1 and nxt[hop] != s:
                hop = nxt[hop]
            table[(s, d)] = hop
    return table


class DijkstraRouter:
    """Baseline định tuyến: next-hop theo đường đi ngắn nhất (shortest path)."""

    def __init__(self, env):
        self.table = dijkstra_next_hops(env.topology.adj_matrix)

    def act(self, env, obs, mask):
        node, packet = env.pending
        return int(self.table[(node, packet["dst"])])


def run_policy(env, select_action, n_episodes, seed=100000):
    rows = []
    for i in range(n_episodes):
        obs, mask, done, _ = env.reset(seed=seed + i)
        ep_rew = 0.0
        while not done:
            action = select_action(env, obs, mask)
            obs, mask, reward, done, info = env.step(action)
            ep_rew += reward
        info["episode_reward"] = ep_rew
        rows.append(info)
    return rows


def summarize(rows):
    keys = ["episode_reward", "avg_delay_ms", "packet_loss_rate", "throughput", "arrived"]
    out = {}
    for k in keys:
        values = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(values.mean())
        out[k + "_std"] = float(values.std())
    return out


def make_rl_select(agent):
    def select_action(env, obs, mask):
        return agent.act(obs, mask, greedy=True)
    return select_action


def run_baseline(env_cfg, n_episodes, seed=100000):
    env = RoutingEnv(**env_cfg)
    router = DijkstraRouter(env)
    rows = run_policy(env, router.act, n_episodes, seed=seed)
    return summarize(rows), rows
