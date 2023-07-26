import os
import sys
import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("modelPath", help="Path to the Model", type=str)
parser.add_argument("-t", "--trials", type=int, default=1000, help="Number of episodes to evaluate the model for in each environment.")
parser.add_argument("--dynamic", action='store_true', help="Use Dynamic Obstacles")
parser.add_argument("--obstacles", "-o", type=int, default=None, help="Number of Obstacles to use in the environment.")
parser.add_argument("--sigma", action="store_true", default =False, help="iterates for sigma values")
parser.add_argument("--mu", action="store_true", default=False, help="iterates for mu values")
parser.add_argument('--local', action='store_true', help='Run on Local Machine')
args = parser.parse_args()

if args.local:

    exec_command = f"python ParallelEvaluationPipeline.py {args.modelPath} -t {args.trials} -o {args.obstacles}"
    if not args.dynamic:
        exec_command += " -d False"
    if args.mu:
        exec_command += " -mu True"
    if args.sigma:
        exec_command += " -si True"

    os.system(exec_command)

else:
    taskName = f"{args.modelPath[-21:-15]}_eval"
    if args.sigma:
        taskName += "_unbiased"
    if args.mu:
        taskName += "_biased"

    with open('evalScriptTemplate.sh', 'r') as f:
        script = ''.join(f.readlines())
    
    script = script.replace("{outputFile}", f"jobOutputs/{taskName}_output.txt")
    script = script.replace("{jobName}", f"{taskName}")
    script = script.replace("{modelPath}", str(args.modelPath))
    script = script.replace("{trials}", str(args.trials))
    script = script.replace("{obstacles}", str(args.obstacles))
    script = script.replace("{dynamic}", str(args.dynamic))
    script = script.replace("{mu}", str(args.mu))
    script = script.replace("{sigma}", str(args.sigma))

    tmp = tempfile.NamedTemporaryFile()

    with open(tmp.name, 'w') as f:
        f.write(script)

    print(f"Dispatching Train Job for {taskName}")

    os.system(f"sbatch {tmp.name}")
