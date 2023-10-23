#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -p cpu_dev
#SBATCH -J exemplo
#SBATCH --exclusive

echo $SLUM_JOB_NODELIST

cd /scratch/pex1272-ufersa/bruno.silva4/teste1

module load gcc/9.3

g++ -g -o teste.exe -fopenmp onda.cpp

./teste.exe 2