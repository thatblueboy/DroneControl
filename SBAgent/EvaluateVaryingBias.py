import sys
import argparse
import numpy as np
import time
import datetime
from tabulate import tabulate
from EvaluateModel import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("modelPath", help="Path to the Model", type=str)
parser.add_argument("-t", "--trials", type=int, default=1000, help="Number of episodes to evaluate the model for in each environment.")
args = parser.parse_args()

mu = 0  # Doesn't matter
sigmas = np.arange(0, 3.1, 0.1)
denoisers = ['None', 'LPF', 'KF']

with open("output.md", "w") as md_file:
    print("# Analysis", file=md_file)
    print(f"**Model**: `{args.modelPath}`", file=md_file)

    for sigma in sigmas:
        for denoiser in denoisers:
            print(f"### $\mu = {mu}$ | $\sigma = {sigma}$ | Denoiser = `{denoiser}`\n", file=md_file)
            print(f"Evaluating {mu, sigma, denoiser}", file=sys.stderr)
            startTime = time.time()
            evaluationTable = evaluate(mu, sigma, denoiser.lower(), args.modelPath, args.trials, gui = True, fixed = False, varyingBias = True, randomizeBiasDirection = True)
            endTime = time.time()
            print(file=sys.stderr)
            print(f"Time Taken: {datetime.timedelta(seconds=endTime - startTime)}", file=sys.stderr)
            print(tabulate(evaluationTable, headers=["Metric", "Value"], tablefmt='github'), file=md_file)
            print("---\n", file=md_file)
