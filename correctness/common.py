import argparse
from functools import partial
import json
import os
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm


def add_common_other_args_and_parse(parser: argparse.ArgumentParser):
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=[
            "vllm",
        ],
    )
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--model-path",
                        type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    args = parser.parse_args()

    if args.port is None:
        default_port = {
            "vllm": 8000,
        }
        args.port = default_port.get(args.backend, None)
    return args


def call_generate_vllm(prompt,
                       temperature,
                       max_tokens,
                       stop=None,
                       n=1,
                       url=None):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    if n == 1:
        pred = res.json()["text"][0][len(prompt):]
    else:
        pred = [x[len(prompt):] for x in res.json()["text"]]
    return pred


def _get_call_generate(args: argparse.Namespace):
    return partial(call_generate_vllm, url=f"{args.host}:{args.port}/generate")


def get_call_generate(args: argparse.Namespace):
    call_generate = _get_call_generate(args)

    def func(*args, **kwargs):
        try:
            return call_generate(*args, **kwargs)
        except Exception as e:
            print("Exception in call_generate:\n" + e)
            raise

    return func


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def dump_state_text(filename: str, states: list, mode: str = "w"):
    """Dump program state in a text file."""
    with open(filename, mode) as fout:
        for i, s in enumerate(states):
            if isinstance(s, str):
                pass
            else:
                s = str(s)

            fout.write("=" * 40 + f" {i} " + "=" * 40 + "\n" + s + "\n" +
                       "=" * 80 + "\n\n")


def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def call_select_vllm(context, choices, url=None):
    assert url is not None

    scores = []
    for i in range(len(choices)):
        data = {
            "prompt": context + choices[i],
            "max_tokens": 1,
            "prompt_logprobs": 1,
        }
        res = requests.post(url, json=data)
        assert res.status_code == 200
        scores.append(res.json().get("prompt_score", 0))
    return np.argmax(scores)


def _get_call_select(args: argparse.Namespace):
    return partial(call_select_vllm, url=f"{args.host}:{args.port}/generate")


def get_call_select(args: argparse.Namespace):
    call_select = _get_call_select(args)

    def func(*args, **kwargs):
        try:
            return call_select(*args, **kwargs)
        except Exception as e:
            print("Exception in call_select:\n" + e)
            raise

    return func
