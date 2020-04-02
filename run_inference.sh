#!/bin/bash
#PBS -N test
#PBS -l nodes=gpu1:ppn=1:gpus=1
#PBS -l pmem=8gb
#PBS -l walltime=00:10:00
#PBS -q gpgpu
#PBS -e /home/Mikhail.Sidorenko/logs/errors1.txt
#PBS -o /home/Mikhail.Sidorenko/logs/output1.txt

cd $PBS_O_WORKDIR
singularity instance start --nv pytorch_19.04-py3 pytorch
singularity shell instance://pytorch cd digital-rock && source env2/bin/activate && python mkldnn_test.py
singularity instance.stop pytorch