import argparse
import os
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from typing import Any

import vllm.envs as envs
from simple_term_menu import TerminalMenu
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


class LlmKwargs(dict):

    def __init__(self, args: argparse.Namespace) -> None:
        self.kwargs = {
            'model': args.model,
            'kv_cache_dtype': args.kv_cache_dtype,
            'tensor_parallel_size': args.tensor_parallel_size,
            'dtype': args.dtype,
            'quantization': args.quantization,
            'enforce_eager': args.enforce_eager,
        }
        self.prompt = args.prompt if args.input_len == -1 else [
            0
        ] * args.input_len
        self.batch_size = args.batch_size
        self.sampling_params = SamplingParams(temperature=args.temperature,
                                              top_p=1,
                                              max_tokens=args.max_tokens,
                                              ignore_eos=args.ignore_eos)
        self.prompt = "There is a round table in the middle of the"

    def __setitem__(self, key: str, value: str) -> None:
        self.kwargs[key] = value

    def __str__(self) -> str:
        res = "===================\n"
        for key, value in self.kwargs.items():
            res += f"{key}: {value}\n"
        res += f"Sampling params: {self.sampling_params}\n"
        res += "\n Misc: \n"
        res += f"Prompt: {self.prompt}\n"
        res += f"Batch size: {self.batch_size}\n"
        return res


def select_model(llm_kwargs: LlmKwargs):
    # Create a list of all the subfolders of a folder
    import os
    folder = "/models"
    folders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders = []
    for subfolder in folders:
        subfolders.extend(
            [f.path for f in os.scandir(subfolder) if f.is_dir()])

    folders.extend(subfolders)
    folder_idx = menu(folders)
    llm_kwargs["model"] = folders[folder_idx]


def select_prompt(llm_kwargs: LlmKwargs):
    llm_kwargs.prompt = input("Enter a prompt: ")


def select_batch_size(llm_kwargs: LlmKwargs):
    llm_kwargs.batch_size = int(input("Enter a batch size: "))


def select_max_tokens(llm_kwargs: LlmKwargs):
    llm_kwargs.sampling_params.max_tokens = int(input("Enter max tokens: "))


def select_input_len(llm_kwargs: LlmKwargs):
    llm_kwargs.prompt = [0] * int(input("Enter input length: "))


def select_temperature(llm_kwargs: LlmKwargs):
    llm_kwargs.sampling_params.temperature = float(
        input("Enter temperature: "))


def select_ignore_eos(llm_kwargs: LlmKwargs):
    llm_kwargs.sampling_params.ignore_eos = [False, True][menu([False, True])]


values = {
    "model": select_model,
    "kv_cache_dtype": ["auto", "fp8"],
    "tensor_parallel_size": [1, 2, 4, 8],
    "dtype": ["auto", "float16", "bfloat16"],
    "quantization": ["None", "fp8", "compressed-tensors", "fbgemm-fp8"],
    "enforce_eager": [True, False],
    "prompt": select_prompt,
    "batch_size": select_batch_size,
    "max_tokens": select_max_tokens,
    "input_len": select_input_len,
    "temperature": select_temperature,
    "ignore_eos": select_ignore_eos,
    "Done": None
}


def menu(items):
    terminal_menu = TerminalMenu([str(x) for x in items])
    menu_entry_index = terminal_menu.show()
    if menu_entry_index is None:
        print("Aborted")
        exit(1)
    return menu_entry_index


def interactive(llm_kwargs: LlmKwargs):
    while True:
        selected = menu(list(values.keys()))
        key = list(values.keys())[selected]
        value = values[list(values.keys())[selected]]
        if value is None:
            return
        if callable(value):
            value(llm_kwargs)
        elif isinstance(value, list):
            new_value = type(value[0])(value[menu(value)])
            if new_value == 'None':
                new_value = None
            llm_kwargs[key] = new_value
        print(llm_kwargs)


def recreate_trace(args: argparse.Namespace):
    from rocpd.schema import RocpdSchema
    if envs.VLLM_RPD_PROFILER_DIR is None:
        envs.VLLM_RPD_PROFILER_DIR = os.path.join(os.path.curdir, "trace.rpd")
    try:
        os.remove(envs.VLLM_RPD_PROFILER_DIR)
    except FileNotFoundError:
        pass
    schema = RocpdSchema()
    connection = sqlite3.connect(envs.VLLM_RPD_PROFILER_DIR)
    schema.writeSchema(connection)
    connection.commit()


def main(args: argparse.Namespace):

    @contextmanager
    def rpd_profiler_context():
        from rpdTracerControl import rpdTracerControl as rpd
        llm.start_profile()
        yield
        llm.stop_profile()
        rpd.top_totals()

    llm_args = LlmKwargs(args)
    print(llm_args)
    if args.interactive:
        interactive(llm_args)

    batch_size = llm_args.batch_size

    if args.rpd:
        recreate_trace(args)

    llm = LLM(**llm_args.kwargs)

    start_time = time.perf_counter()
    with rpd_profiler_context() if args.rpd else nullcontext():
        outs = llm.generate([TokensPrompt(prompt_token_ids=llm_args.prompt)] *
                            llm_args.batch_size if isinstance(
                                llm_args.prompt, list) else [llm_args.prompt] *
                            batch_size,
                            sampling_params=llm_args.sampling_params)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    out_lengths = [len(x.token_ids) for out in outs for x in out.outputs]
    num_tokens = sum(out_lengths)

    print(
        f"{num_tokens} tokens. {num_tokens / batch_size} on average. {num_tokens / elapsed_time:.2f} tokens/s. {elapsed_time} seconds"
    )
    for out in outs:
        print("===========")
        print(out.outputs[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Test much')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='Model to use',
                        default='/models/llama-2-7b-chat-hf')
    parser.add_argument('-i',
                        '--interactive',
                        action='store_true',
                        help='Interactive mode')
    parser.add_argument('--kv_cache_dtype',
                        type=str,
                        help='KV Cache Data Type',
                        choices=values["kv_cache_dtype"],
                        default='auto')
    parser.add_argument('-tp',
                        '--tensor-parallel-size',
                        type=int,
                        default=1,
                        choices=values["tensor_parallel_size"])
    parser.add_argument('--dtype',
                        type=str,
                        default='auto',
                        choices=values["dtype"])
    parser.add_argument('--quantization',
                        type=str,
                        default=None,
                        choices=values["quantization"])
    parser.add_argument('--prompt',
                        type=str,
                        default="There is a round table in the middle of the")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--rpd', action='store_true')
    parser.add_argument('--input-len', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--ignore-eos', action='store_true')
    args = parser.parse_args()

    main(args)
