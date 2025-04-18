#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -p a6000
#SBATCH --job-name=3dmedformer
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -o /home/l.cai/job_logs/train_%A_%a.out
#SBATCH -e /home/l.cai/job_logs/train_%A_%a.err


source /home/l.cai/miniconda3/bin/activate base
python train.py --model medformer --dimension 3d --dataset acdc --batch_size 2 --unique_name acdc_3d_medformer_rectal --gpu 0
