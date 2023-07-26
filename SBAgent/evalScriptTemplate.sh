#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --job-name={jobName}
#SBATCH -o {outputFile}

python ParallelEvaluationPipeline.py {modelPath} -t {trials} -d {dynamic} -o {obstacles} -mu {mu} -si {sigma}
