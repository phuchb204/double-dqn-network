import random
from collections import deque


class ReplayBuffer:
    """Bộ đệm replay dạng hàng đợi với lấy mẫu đồng đều."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones), list(masks)

    def __len__(self):
        return len(self.buffer)
