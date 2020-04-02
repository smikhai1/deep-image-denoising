#!/bin/bash
#PBS -N test
#PBS -l nodes=gpu1:ppn=8:gpus=4
#PBS -l pmem=20gb
#PBS -l walltime=08:00:00
#PBS -q gpgpu
#PBS -e /home/Mikhail.Sidorenko/digital-rock/trained-models/red-net/ct/mse-loss/stderr.txt
#PBS -o /home/Mikhail.Sidorenko/digital-rock/trained-models/red-net/ct/mse-loss/stdout.txt

cd $PBS_O_WORKDIR
singularity instance start --nv pytorch_19.04-py3.sif pytorch
cd digital-rock
singularity exec instance://pytorch python3 train.py --config ./configs/config_train.yml --paths ./configs/path.yml
singularity instance stop pytorch