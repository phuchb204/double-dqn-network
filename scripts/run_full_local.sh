#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1

TOPOLOGY=hub
LOADS="2,4,6,8"
TRAIN_LOADS="6.0 8.0"
EPISODES=2000
EVAL_EVERY=200
EVAL_EPISODES=20
LR=1e-3
BATCH=128
GAMMA=0.99
EPS_DECAY=30000
PY=.venv/bin/python

echo "== Baseline Dijkstra (loads: $LOADS) =="
$PY experiments/baseline.py --topology $TOPOLOGY --loads $LOADS --episodes 50

for load in $TRAIN_LOADS; do
  echo "== Load $load: DQN + Double DQN (song song) =="
  $PY experiments/train_dqn.py --topology $TOPOLOGY --load $load \
      --episodes $EPISODES --eval-every $EVAL_EVERY --eval-episodes $EVAL_EPISODES \
      --lr $LR --batch-size $BATCH --gamma $GAMMA --eps-decay-steps $EPS_DECAY \
      --out-dir results > "results/dqn_load${load}.log" 2>&1 &
  $PY experiments/train_ddqn.py --topology $TOPOLOGY --load $load \
      --episodes $EPISODES --eval-every $EVAL_EVERY --eval-episodes $EVAL_EPISODES \
      --lr $LR --batch-size $BATCH --gamma $GAMMA --eps-decay-steps $EPS_DECAY \
      --out-dir results > "results/ddqn_load${load}.log" 2>&1 &
  wait
done

echo "== Plot results =="
$PY experiments/plot_results.py
echo "DONE"
