from typing import Any
from simple_term_menu import TerminalMenu
import argparse
import time

from vllm import LLM, SamplingParams


class LlmKwargs(dict):

    def __init__(self):
        self.kwargs = {}
        self.batch_size = 1
        self.max_tokens = 256
        self.prompt = "There is a round table in the middle of the"

    def from_args(self, args: argparse.Namespace):
        self.kwargs['model'] = args.model
        self.kwargs['kv_cache_dtype'] = args.kv_cache_dtype
        self.kwargs['tensor_parallel_size'] = args.tensor_parallel_size
        self.kwargs['dtype'] = args.dtype
        self.kwargs['quantization'] = args.quantization
        self.kwargs['enforce_eager'] = args.enforce_eager
        self.prompt = args.prompt
        self.batch_size = args.batch_size
        self.max_tokens = args.max_tokens

    def __setitem__(self, key: str, value: str) -> None:
        self.kwargs[key] = value

    def print(self):
        for key, value in self.kwargs.items():
            print(f"{key}: {value}")

    def __str__(self) -> str:
        res = "===================\n"
        for key, value in self.kwargs.items():
            res += f"{key}: {value}\n"
        res += "\n Sampling params: \n"
        res += f"Prompt: {self.prompt}\n"
        res += f"Batch size: {self.batch_size}\n"
        res += f"Max tokens: {self.max_tokens}\n"
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
    llm_kwargs.max_tokens = int(input("Enter max tokens: "))


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
    "Done": None
}


def menu(items):
    terminal_menu = TerminalMenu([str(x) for x in items])
    menu_entry_index = terminal_menu.show()
    if menu_entry_index is None:
        print("Abroted")
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


def main(args: argparse.Namespace):
    llm_args = LlmKwargs()
    llm_args.from_args(args)
    print(llm_args)
    if args.interactive:
        interactive(llm_args)

    batch_size = llm_args.batch_size
    llm = LLM(**llm_args.kwargs)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=llm_args.max_tokens)
    start_time = time.perf_counter()
    outs = llm.generate([llm_args.prompt] * batch_size,
                        sampling_params=sampling_params)
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
    parser.add_argument('--prompt', type=str, default="There is a round table in the middle of the")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args()

    main(args)
