# Double DQN cho định tuyến trong mạng máy tính

Thực nghiệm tốt nghiệp: **"Tìm hiểu và cài đặt thử nghiệm giải thuật học tăng cường
sâu Double DQN ứng dụng trong mạng máy tính."**

So sánh **DQN**, **Double DQN** và baseline **Dijkstra (shortest-path)** trên bài toán
định tuyến theo next-hop với các chỉ số: phần thưởng, độ trễ, tỷ lệ mất gói, thông lượng.

## Mục lục

- [Cấu trúc project](#cấu-trúc-project)
- [Mô hình bài toán](#mô-hình-bài-toán)
- [Cài đặt](#cài-đặt)
- [Chạy thực nghiệm](#chạy-thực-nghiệm)
- [Hyperparameter](#hyperparameter)
- [Kết quả](#kết-quả)
- [Tái hiện trên Colab](#tái-hiện-trên-colab)

## Cấu trúc project

```
TTTN/
├── src/
│   ├── environment.py      # Network simulator + môi trường định tuyến (gym-like)
│   ├── replay_buffer.py    # Replay buffer lấy mẫu đồng đều
│   ├── dqn.py              # QNetwork + DQNAgent (DQN & Double DQN)
│   ├── train.py            # Vòng huấn luyện, log CSV, checkpoint model
│   └── evaluate.py         # Đánh giá policy + baseline Dijkstra
├── experiments/
│   ├── train_dqn.py        # Huấn luyện DQN (CLI)
│   ├── train_ddqn.py       # Huấn luyện Double DQN (CLI)
│   ├── baseline.py         # Chạy baseline trên nhiều mức tải
│   └── _common.py          # Tham số CLI dùng chung
├── notebooks/
│   ├── plots.ipynb         # Vẽ biểu đồ từ results/csv
│   └── analysis.ipynb      # Phân tích kết quả
├── results/
│   ├── csv/                # Log kết quả huấn luyện / baseline
│   ├── models/             # Checkpoint
│   └── figures/            # Biểu đồ xuất ra
├── requirements.txt
└── README.md
```

## Mô hình bài toán

- **Mạng**: đồ thị vô hướng ngẫu nhiên, mỗi node có hàng đợi dung lượng `buffer`
  và tốc độ phục vụ `mu` gói/round.
- **Traffic**: `n_flows` luồng nguồn-đích ngẫu nhiên, mỗi luồng sinh gói theo phân
  bố Bernoulli với xác suất sao cho tổng cường độ tải xấp xỉ `load` gói/round.
- **Agent**: tại mỗi gói đang ở đầu hàng đợi của một node, quan sát trạng thái
  `[one-hot node hiện tại | one-hot node đích | độ dài hàng đợi chuẩn hóa]` và
  chọn **next-hop** hợp lệ (hàng xóm của node hiện tại).
- **Reward**: `+1` gói đến đích, `-1` gói bị drop (queue đầy / vượt `max_hops` /
  action không hợp lệ), `-hop_penalty` mỗi lần chuyển tiếp (ưu tiên đường ngắn).
- **Metrics**: tổng phần thưởng/episode, độ trễ trung bình (ms), tỷ lệ mất gói,
  thông lượng (gói/ms).

**DQN vs Double DQN** (khác biệt duy nhất ở công thức target):

```
DQN:    target = r + gamma * max_a  Q_target(s', a)
Double: target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
```

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU (máy local):
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Chạy thực nghiệm

```bash
# Huấn luyện DQN
python experiments/train_dqn.py --load 0.7 --episodes 2000

# Huấn luyện Double DQN
python experiments/train_ddqn.py --load 0.7 --episodes 2000

# Baseline trên nhiều mức tải
python experiments/baseline.py --loads 0.3,0.5,0.7,0.9,1.1 --episodes 50
```

Kết quả được lưu vào `results/csv/` (log CSV), `results/models/` (checkpoint).

Vẽ biểu đồ:

```bash
python experiments/plot_results.py          # tạo figure từ results/csv
# hoặc mở notebooks/plots.ipynb trên Colab/Jupyter
```

## Hyperparameter

| Tham số            | CLI                      | Mặc định |
| ------------------ | ------------------------ | -------- |
| Số node            | `--n-nodes`              | 10       |
| Dung lượng queue   | `--buffer`               | 8        |
| Tốc độ phục vụ     | `--mu`                   | 3        |
| Cường độ tải       | `--load`                 | 0.7      |
| Số luồng           | `--n-flows`              | 8        |
| Số round/episode   | `--total-rounds`         | 40       |
| Số episode         | `--episodes`             | 1000     |
| Learning rate      | `--lr`                   | 1e-3     |
| Batch size         | `--batch-size`           | 128      |
| Discount factor γ  | `--gamma`                | 0.99     |
| Hidden units       | `--hidden`               | 128      |
| ε start / end      | `--eps-start`/`--eps-end`| 1.0/0.05 |
| ε decay steps      | `--eps-decay-steps`      | 20000    |
| Target update freq | `--target-update-freq`   | 500      |
| Eval interval      | `--eval-every`           | 100      |

## Kết quả

Các tệp CSV trong `results/csv/` gồm các cột: `episode, train_steps,
train_reward, train_loss, eps, train_avg_delay_ms, train_loss_rate,
train_throughput, eval_reward, eval_avg_delay_ms, eval_loss_rate,
eval_throughput`. Baseline ghi trong `baseline.csv` theo từng mức `load`.

## Tái hiện trên Colab

```python
!git clone <repo-url> double-dqn-network && cd double-dqn-network
!pip install -r requirements.txt
!python experiments/train_ddqn.py --load 0.7 --episodes 5000
```

Sau đó dùng `notebooks/plots.ipynb` để vẽ biểu đồ từ `results/csv/`.
