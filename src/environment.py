import random

import numpy as np


class NetworkTopology:
    """Sinh đồ thị mạng và ma trận kề.

    topology='random': đồ thị ngẫu nhiên liên thông, bậc giới hạn.
    topology='hub':    hub-and-spoke (node 0 là trung tâm, các node khác nối
                       vòng với nhau) — tạo bottleneck tự nhiên để agent học
                       né tắc nghẽn bằng đường vòng dài hơn.
    """

    def __init__(self, n_nodes, seed=0, max_degree=4, topology="random"):
        self.n_nodes = n_nodes
        self.seed = seed
        self.topology = topology
        rng = random.Random(seed)
        self.adj = [set() for _ in range(n_nodes)]

        if topology == "hub":
            self._build_hub(rng, max_degree)
        else:
            self._build_random(rng, max_degree)

        self.adj = [sorted(s) for s in self.adj]
        self.adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for u in range(n_nodes):
            for v in self.adj[u]:
                self.adj_matrix[u, v] = 1.0

    def _build_random(self, rng, max_degree):
        edges = set()
        for u in range(self.n_nodes):
            targets = list(range(self.n_nodes))
            targets.remove(u)
            rng.shuffle(targets)
            for v in targets:
                if len(self.adj[u]) >= max_degree:
                    break
                if v in self.adj[u] or len(self.adj[v]) >= max_degree:
                    continue
                if (u, v) in edges or (v, u) in edges:
                    continue
                self.adj[u].add(v)
                self.adj[v].add(u)
                edges.add((u, v))
                if len(self.adj[u]) >= max_degree:
                    break
        self._ensure_connected(rng)

    def _build_hub(self, rng, max_degree):
        hub = 0
        leaves = list(range(1, self.n_nodes))
        for leaf in leaves:
            self.adj[hub].add(leaf)
            self.adj[leaf].add(hub)
        if len(leaves) >= 3:
            for i in range(len(leaves)):
                a = leaves[i]
                b = leaves[(i + 1) % len(leaves)]
                self.adj[a].add(b)
                self.adj[b].add(a)
        candidates = [(a, b) for a in leaves for b in leaves
                      if a < b and b not in self.adj[a] and a != hub and b != hub]
        rng.shuffle(candidates)
        for a, b in candidates:
            if len(self.adj[a]) >= max_degree or len(self.adj[b]) >= max_degree:
                continue
            self.adj[a].add(b)
            self.adj[b].add(a)

    def _ensure_connected(self, rng):
        visited = {0}
        frontier = list(self.adj[0])
        while frontier and len(visited) < self.n_nodes:
            u = frontier.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in self.adj[u]:
                if v not in visited:
                    frontier.append(v)
        for node in range(self.n_nodes):
            if node not in visited:
                partner = rng.choice(sorted(visited))
                self.adj[node].add(partner)
                self.adj[partner].add(node)
                visited.add(node)

    def neighbors(self, node):
        return self.adj[node]

    def is_neighbor(self, u, v):
        return self.adj_matrix[u, v] == 1.0


class RoutingEnv:
    """Môi trường định tuyến theo next-hop (discrete-event round-based).

    Mỗi round: (1) sinh gói mới từ các flow theo phân bố Bernoulli, (2) mỗi node
    xử lý tối đa `mu` gói ở đầu hàng đợi — agent quyết định next hop cho từng gói.
    Queue đầy -> drop. Gói đến đích -> hoàn thành.
    """

    def __init__(self, n_nodes=10, seed=0, topology="random", buffer=8, mu=3,
                 load=0.7, n_flows=8, total_rounds=40, time_per_round=5.0,
                 max_hops=None, hop_penalty=0.02, drop_reward=-1.0,
                 arrive_reward=1.0, invalid_reward=-1.0):
        self.topology_obj = NetworkTopology(n_nodes, seed=seed, topology=topology)
        self.topology = self.topology_obj
        self.n_nodes = n_nodes
        self.topology_name = topology
        self.buffer = buffer
        self.mu = mu
        self.load = load
        self.n_flows = n_flows
        self.total_rounds = total_rounds
        self.time_per_round = time_per_round
        self.max_hops = max_hops or 4 * n_nodes
        self.hop_penalty = hop_penalty
        self.drop_reward = drop_reward
        self.arrive_reward = arrive_reward
        self.invalid_reward = invalid_reward
        self._rng = random.Random(0)

        self.state_dim = 3 * n_nodes
        self.action_dim = n_nodes
        self._init_flows()

    def _init_flows(self):
        if self.topology_name == "hub":
            pool = list(range(1, self.n_nodes))
        else:
            pool = list(range(self.n_nodes))
        pairs = [(s, d) for s in pool for d in pool if s != d]
        rng = random.Random(0)
        rng.shuffle(pairs)
        self.flow_pairs = pairs[: self.n_flows]
        self.flow_probs = [self.load / self.n_flows for _ in self.flow_pairs]

    def _onehot(self, idx):
        vec = np.zeros(self.n_nodes, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _queue_state(self):
        q = np.array([len(qq) for qq in self.queues], dtype=np.float32)
        return q / max(1, self.buffer)

    def _make_obs(self):
        if self.pending is None:
            return None, None
        node, packet = self.pending
        obs = np.concatenate([
            self._onehot(node),
            self._onehot(packet["dst"]),
            self._queue_state(),
        ]).astype(np.float32)
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for v in self.topology.neighbors(node):
            mask[v] = 1.0
        return obs, mask

    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        self.t = 0
        self.queues = [[] for _ in range(self.n_nodes)]
        self.sent = 0
        self.dropped = 0
        self.aborted = 0
        self.arrived = 0
        self.delay_samples = []
        self.pending = None
        self._round_items = []
        self._round_idx = 0
        self._episode_reward = 0.0

        self._advance()
        obs, mask = self._make_obs()
        if obs is None:
            self.done = True
            return None, None, True, self._info()
        self.done = False
        return obs, mask, False, {}

    def _start_round(self):
        items = []
        for node in range(self.n_nodes):
            take = min(self.mu, len(self.queues[node]))
            for k in range(take):
                items.append((node, self.queues[node][k]))
        self._round_items = items
        self._round_idx = 0

    def _advance(self):
        while self.pending is None and self.t < self.total_rounds:
            if self._round_idx >= len(self._round_items):
                self._arrivals()
                self._start_round()
                self.t += 1
                if self._round_idx < len(self._round_items):
                    self.pending = self._round_items[self._round_idx]
                    return
                continue
            self.pending = self._round_items[self._round_idx]
            return

    def _arrivals(self):
        for (src, dst), p in zip(self.flow_pairs, self.flow_probs):
            if self._rng.random() < p:
                if len(self.queues[src]) >= self.buffer:
                    self.dropped += 1
                else:
                    self.sent += 1
                    self.queues[src].append({
                        "dst": dst,
                        "hops": 0,
                        "t_birth": self.t,
                        "t_arrive": None,
                    })

    def step(self, action):
        assert not self.done, "episode da ket thuc"
        node, packet = self.pending
        self._round_idx += 1
        assert len(self.queues[node]) > 0 and self.queues[node][0] is packet
        self.queues[node].pop(0)
        action = int(action)
        r = 0.0

        if packet["hops"] >= self.max_hops:
            self.aborted += 1
            r = self.drop_reward
        elif not self.topology.is_neighbor(node, action):
            self.dropped += 1
            r = self.invalid_reward
        elif action == packet["dst"]:
            packet["t_arrive"] = self.t
            packet["hops"] += 1
            self.arrived += 1
            self.delay_samples.append((self.t - packet["t_birth"]) * self.time_per_round)
            r = self.arrive_reward
        else:
            packet["hops"] += 1
            if len(self.queues[action]) >= self.buffer:
                self.dropped += 1
                r = self.drop_reward
            else:
                self.queues[action].append(packet)
                r = -self.hop_penalty

        self._episode_reward += r
        self.pending = None
        self._advance()

        obs, mask = self._make_obs()
        if obs is None:
            self.done = True
            return None, None, r, True, self._info()
        return obs, mask, r, False, {}

    def _info(self):
        in_flight = sum(len(q) for q in self.queues)
        total_sent = self.sent + self.dropped
        loss_rate = self.dropped / total_sent if total_sent else 0.0
        throughput = self.arrived / (self.total_rounds * self.time_per_round)
        avg_delay = float(np.mean(self.delay_samples)) if self.delay_samples else 0.0
        return {
            "episode_reward": self._episode_reward,
            "sent": self.sent,
            "dropped": self.dropped,
            "aborted": self.aborted,
            "arrived": self.arrived,
            "in_flight": in_flight,
            "packet_loss_rate": loss_rate,
            "throughput": throughput,
            "avg_delay_ms": avg_delay,
        }
