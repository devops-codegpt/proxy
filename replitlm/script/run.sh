#!/bin/sh
# This script is used to run the inference of Replit.

GPU="$1"

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

CHECKPOINT_PATH="./model/replit-code-v1-3b/"

# export CUDA settings
if [ -z "$GPU" ]; then
  DEVICE="cpu"
else
  DEVICE="cuda:$GPU"
fi

export CUDA_HOME=/usr/local/cuda-11.7/

# remove --greedy if using sampling
CMD="python $MAIN_DIR/src/main.py \
        --checkpoint $CHECKPOINT_PATH \
        --out-seq-length 128 \
        --num-return-sequences 1 \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 0 \
        --device $DEVICE \
        --do-sample"

echo "$CMD"
eval "$CMD"