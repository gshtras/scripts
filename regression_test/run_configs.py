import argparse
import json
import subprocess
import time
import psycopg2
import os
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime


def find_projects():
    projects_path = os.path.join(os.path.expanduser("~"), "Projects")
    if os.path.exists(projects_path):
        return projects_path
    projects_path = os.path.join(os.path.abspath(os.sep), "projects")
    if os.path.exists(projects_path):
        return projects_path
    raise Exception("Projects path not found")


load_dotenv()
connection = psycopg2.connect(database=os.getenv("DASHBOARD_DATABASE"),
                              user=os.getenv("DASHBOARD_USER"),
                              password=os.getenv("DASHBOARD_PASSWORD"),
                              host=os.getenv("DASHBOARD_HOST"),
                              port=os.getenv("DASHBOARD_PORT"))

cursor = connection.cursor()
today_date = datetime.now().strftime("%Y-%m-%d")
vllm_version = ""


def run_command(command, env, timeout=30 * 60):
    env_dict = dict(item.split('=') for item in env) if '=' in env else {}
    output = subprocess.Popen(command, shell=True, env=os.environ | env_dict)
    try:
        output.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        output.terminate()
        print("Run timed out")
        return False
    if output.returncode != 0:
        print("Run failed")
        return False
    else:
        print("Run succeeded")
        return True

def get_version():
    global vllm_version
    output = subprocess.run("pip show vllm | grep Version:", shell=True, stdout=subprocess.PIPE)
    vllm_version = output.stdout.strip()

def cleanup():
    run_command("ps -alef | grep benchmark | awk '{print $4}' | xargs kill -9", "")
    time.sleep(5)

def getEnabledConfigs(args: argparse.Namespace):
    if args.short_run:
        return "WHERE enabled = 'SHORT_RUN'"
    return "WHERE enabled != 'NONE'"

def run_performance(args: argparse.Namespace):
    print("Starting performance runs")
    cursor.execute(f'''SELECT id, dtype, "batchSize", "inputLength", "outputLength", tp, "modelName", environment, "extraParams" FROM "PerformanceConfig" {getEnabledConfigs(args)}''')
    data = cursor.fetchall()
    for row in tqdm(data):
        id, dtype, batch, input, output, tp, model, env, extra = row
        cursor.execute('''select path from "Model" where name = %s''', (model,))
        model_path = cursor.fetchone()[0]
        benchmark_path = f"{os.path.join(args.vllm_path, 'benchmarks', 'benchmark_latency.py')}"
        command = f"python {benchmark_path} --model /models/{model_path} --dtype {dtype} --batch-size {batch} --input-len {input} --output-len {output} -tp {tp} {' '.join(extra)}"
        if output == 1:
            command += " --enforce-eager"
        else:
            command += " --num-scheduler-steps 10"
        command += f" --output-json /projects/tmp/{model}.json --load-format dummy --num-iters-warmup 2 --num-iters 5"
        print(command, flush=True)
        res = run_command(command, env)
        if not res:
            continue
        j = json.load(open(f"/projects/tmp/{model}.json"))
        cursor.execute('''INSERT INTO "PerformanceResult" ("performanceConfigId", "createdAt", latency, "vllmVersion") VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING''', (id, today_date, j['avg_latency'], vllm_version))
        cursor.execute('COMMIT')
        cleanup()

def run_correctness(args: argparse.Namespace):
    print("Starting correctness runs")
    cursor.execute(f'''SELECT id, dtype, "modelName", environment, "extraParams" FROM "CorrectnessConfig" {getEnabledConfigs(args)}''')
    data = cursor.fetchall()
    for row in tqdm(data):
        id, dtype, model, env, extra = row
        cursor.execute('''select path from "Model" where name = %s''', (model,))
        model_path = cursor.fetchone()[0]
        command = f"python /projects/llm_test.py --model /models/{model_path} --dtype {dtype} {' '.join(extra)}"
        command += f" --output-json /projects/tmp/{model}.json"
        print(command, flush=True)
        res = run_command(command, env)
        if not res:
            continue
        j = json.load(open(f"/projects/tmp/{model}.json"))
        cursor.execute('''INSERT INTO "CorrectnessResult" ("correctnessConfigId", "createdAt", generated, "vllmVersion") VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING''', (id, today_date, j['generated'], vllm_version))
        cursor.execute('COMMIT')
        cleanup()

def run_p3l(args: argparse.Namespace):
    print("Starting P3L runs")
    cursor.execute(f'''SELECT id, dtype, "modelName", "contextLen", "sampleSize", "patchSize", environment, "extraParams" FROM "P3LConfig" {getEnabledConfigs(args)}''')
    data = cursor.fetchall()
    for row in tqdm(data):
        id, dtype, model, contextLen, sampleSize, patchSize, env, extra = row
        cursor.execute('''select path from "Model" where name = %s''', (model,))
        model_path = cursor.fetchone()[0]
        benchmark_path = f"{os.path.join(args.vllm_path, 'benchmarks', 'P3L.py')}"
        command = f"python {benchmark_path} --model /models/{model_path} --dtype {dtype} --context-size {contextLen} --sample-size {sampleSize} --patch-size {patchSize} {' '.join(extra)}"
        command += f" --output-json /projects/tmp/{model}.json"
        print(command, flush=True)
        res = run_command(command, env)
        if not res:
            continue
        j = json.load(open(f"/projects/tmp/{model}.json"))
        cursor.execute('''INSERT INTO "P3LResult" ("p3lConfigId", "createdAt", "P3L", "vllmVersion") VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING''', (id, today_date, j['ppl'], vllm_version))
        cursor.execute('COMMIT')
        cleanup()



def main(args: argparse.Namespace):
    get_version()
    cleanup()
    run_correctness(args)
    run_performance(args)
    run_p3l(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regression tests")
    parser.add_argument("--vllm-path", type=str, required=True)
    parser.add_argument("--short-run", action="store_true")
    args = parser.parse_args()
    main(args)