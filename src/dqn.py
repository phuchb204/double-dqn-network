import copy

import numpy as np
import torch
import torch.nn as nn

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Mạng Q: state -> Q-value cho từng action (next-hop)."""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Agent DQN / Double DQN với replay buffer, target network, epsilon-greedy.

    Khác biệt DQN vs Double DQN nằm ở cách tính target:
      - DQN:      target = r + gamma * max_a  Q_target(s', a)
      - Double:   target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
    """

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        self.online = QNetwork(state_dim, action_dim, config.get("hidden", 128)).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.get("lr", 1e-3))
        self.buffer = ReplayBuffer(config.get("buffer_capacity", 50000))
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 128)
        self.use_double = config.get("use_double", False)
        self.target_update_freq = config.get("target_update_freq", 500)
        self.tau = config.get("tau", 1.0)
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.05)
        self.eps_decay_steps = config.get("eps_decay_steps", 10000)

        self.steps_done = 0
        self._rng = np.random.default_rng(config.get("seed", 0))

    @property
    def epsilon(self):
        frac = min(1.0, self.steps_done / max(1, self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def _masked_q(self, model, states, masks):
        q = model(states)
        mask_t = torch.as_tensor(np.asarray(masks, dtype=np.float32), device=self.device)
        return q * mask_t - (1.0 - mask_t) * 1e9

    def act(self, state, mask, greedy=False):
        state = np.asarray(state, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        valid = np.flatnonzero(mask == 1.0)
        if len(valid) == 0:
            return 0
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.choice(valid))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).cpu().numpy()[0]
        q = q * mask - (1.0 - mask) * 1e9
        return int(np.argmax(q))

    def store(self, state, action, reward, next_state, done, mask):
        if next_state is None:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        if mask is None:
            mask = np.zeros(self.action_dim, dtype=np.float32)
        self.buffer.push(state, action, reward, next_state, done, mask)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones, masks = self.buffer.sample(self.batch_size)

        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.online(s).gather(1, a)

        with torch.no_grad():
            if self.use_double:
                best_a = self._masked_q(self.online, s2, masks).argmax(1, keepdim=True)
                q2 = self.target(s2).gather(1, best_a)
            else:
                q2 = self._masked_q(self.target, s2, masks).max(1, keepdim=True).values
            target = r + self.gamma * (1.0 - d) * q2

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self._update_target()
        return float(loss.item())

    def _update_target(self):
        if self.tau >= 1.0:
            self.target.load_state_dict(self.online.state_dict())
        else:
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def save(self, path):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "config": self.config,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
