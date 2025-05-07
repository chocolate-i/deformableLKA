#!/bin/sh

DATASET_PATH=/root/chennuo/deformableLKA/3D/DATASET

# 修改为CUDA 10.2环境设置
export CUDA_HOME=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

export PYTHONPATH=./
export RESULTS_FOLDER=output_synapse_test_continuing
export d_lka_former_preprocessed="$DATASET_PATH"/d_lka_former_raw/d_lka_former_raw_data/Task02_Synapse
export d_lka_former_raw_data_base="$DATASET_PATH"/d_lka_former_raw

python d_lka_former/run/run_training.py 3d_fullres d_lka_former_trainer_synapse 2 0 --continue_training --trans_block TransformerBlock  --depths 3 --skip_connections 4