#! /usr/bin/env bash
#SBATCH --job-name=GEMM_BENCHMARKS
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load cuda intel-oneapi-mkl

OUTPUT="gemm_benchmarks.jld2"
SIZES=()
for n in seq $1 $2
    SIZES+=$((2 ** n))
end

julia gemm_benchmarks.jl "${OUTPUT}" "${SIZES}"
