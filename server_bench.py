import requests
import multiprocessing as mp
import time
import json
import random

def process_request(id, r, start_time, q):
    stats = {
        "prefil": 0,
        "total": 0,
        "tokens": 0,
    }
    for line in r.iter_lines():
        if line:
            stats["tokens"] += 1
            if not stats["prefil"]:
                stats["prefil"] = time.time() - start_time
            #print(f"{id}: {line}")
    stats["total"] = time.time() - start_time
    print(f"Finished: {id}")
    q.put((id, stats))

prompts = []

def load_json():
    global prompts
    prompts = json.load(open("/models/ShareGPT_V3_unfiltered_cleaned_split.json"))

def send_request():
    prompt = prompts[random.randint(0, len(prompts)-1)]['conversations'][0]['value']
    start_time = time.time()
    r = requests.post('http://localhost:8000/v1/completions',
                      headers={
                          "Content-Type": "application/json",
                      },
                      json={
                          "prompt": prompt,
                          "max_tokens": 1024,
                          "ignore_eos": True,
                          "model": "/models/llama-2-70b-chat-hf",
                          "temperature": 0,
                          "top_p": 0.95,
                          "stream": True
                      },
                      stream=True)
    return r, start_time

class Runner:
    def __init__(self) -> None:
        self.processes = []
        self.request_id = 0
        self.q = mp.Queue()
        self.sent_requests = 0
    
    def process_stats(self, id, stats):
        print(f"Request: {id} prefill time: {stats['prefil']*1000}ms total time: {stats['total']*1000}ms tk/s: {stats['tokens']/stats['total']}")

    def prune_requests(self):
        for r in self.processes:
            if not r.is_alive():
                self.processes.remove(r)
                (id, stats) = self.q.get()
                self.process_stats(id, stats)

    def run(self):
        while True:
            print(f"Running: {len(self.processes)}")
            self.prune_requests()
            if self.sent_requests < 100 and len(self.processes) < 50:
                self.add_request()
            if len(self.processes) == 0:
                break
            time.sleep(0.3)

    def add_request(self):
        r, start_time = send_request()
        self.sent_requests += 1
        p = mp.Process(target=process_request, args=(self.request_id, r, start_time, self.q))
        self.request_id += 1
        self.processes.append(p)
        p.start()

    def stop(self):
        for p in self.processes:
            p.join()


def main():
    load_json()
    runner = Runner()
    runner.run()
    runner.stop()


if __name__ == '__main__':
    main()
