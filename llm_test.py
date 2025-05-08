import argparse
import dataclasses
import os
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from typing import Any

import vllm.envs as envs
from vllm.engine.arg_utils import EngineArgs
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.utils import FlexibleArgumentParser
from PIL import Image

class LlmKwargs(dict):

    def __init__(self, args: argparse.Namespace) -> None:
        self.engine_args = EngineArgs.from_cli_args(args)
        self.prompt = args.prompt if args.input_len == -1 else [
            0
        ] * args.input_len
        self.batch_size = args.batch_size
        self.sampling_params = SamplingParams(temperature=args.temperature,
                                              top_p=1,
                                              max_tokens=args.max_tokens,
                                              ignore_eos=args.ignore_eos)
        self.rpd = args.rpd
        try:
            self.rpd_path = envs.VLLM_RPD_PROFILER_DIR
        except:
            self.rpd_path = None
        if self.rpd_path is None:
            self.rpd_path = args.rpd_path or os.path.join(
                os.path.curdir, "trace.rpd")
        self.image_path = args.image_path
        self.serverlike = args.serverlike


    def set_rpd(self, value: str):
        self.rpd = value
        if self.rpd and self.rpd_path is None:
            self.rpd_path = os.path.join(os.path.curdir, "trace.rpd")

    def __setitem__(self, key: str, value: str) -> None:
        self.engine_args = dataclasses.replace(self.engine_args, **{key: value})

    def __str__(self) -> str:
        res = "\n === Engine Args === \n"
        res += f"{self.engine_args}\n"
        res += "\n === Sampling params ===\n"
        res += f"{self.sampling_params}\n"
        res += "\n === Misc === \n"
        res += f"Prompt: {self.prompt}\n"
        res += f"Batch size: {self.batch_size}\n"
        res += f"RPD: {self.rpd}: {self.rpd_path}\n"
        res += f"Image path: {self.image_path}\n"
        res += f"Serverlike: {self.serverlike}\n"
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
    folders = [x for x in folders if ".cache" not in x]
    folder_idx = menu(folders)
    llm_kwargs.engine_args.model = folders[folder_idx]


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


def select_rpd(llm_kwargs: LlmKwargs):
    llm_kwargs.set_rpd([False, True][menu([False, True])])


def select_rpd_path(llm_kwargs: LlmKwargs):
    llm_kwargs.rpd_path = input("Enter RPD path: ")


def select_image_path(llm_kwargs: LlmKwargs):
    llm_kwargs.image_path = input("Enter image path: ")


def select_serverlike(llm_kwargs: LlmKwargs):
    llm_kwargs.serverlike = [False, True][menu([False, True])]


values = {
    "model": select_model,
    "kv_cache_dtype": ["auto", "fp8"],
    "tensor_parallel_size": [1, 2, 4, 8],
    "dtype": ["auto", "float16", "bfloat16"],
    "quantization": ["None", "fp8", "compressed-tensors", "fbgemm-fp8"],
    "enforce_eager": [True, False],
    "disable_custom_all_reduce": [False, True],
    "num_scheduler_steps": [1, 10],
    "prompt": select_prompt,
    "batch_size": select_batch_size,
    "max_tokens": select_max_tokens,
    "input_len": select_input_len,
    "temperature": select_temperature,
    "ignore_eos": select_ignore_eos,
    "rpd": select_rpd,
    "rpd_path": select_rpd_path,
    "image_path": select_image_path,
    "serverlike": select_serverlike,
    "Done": None
}


def menu(items):
    from simple_term_menu import TerminalMenu
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


def recreate_trace(llm_args: LlmKwargs):
    from rocpd.schema import RocpdSchema
    if envs.VLLM_RPD_PROFILER_DIR != llm_args.rpd_path:
        envs.VLLM_RPD_PROFILER_DIR = llm_args.rpd_path
    try:
        os.remove(llm_args.rpd_path)
    except FileNotFoundError:
        pass
    schema = RocpdSchema()
    connection = sqlite3.connect(llm_args.rpd_path)
    schema.writeSchema(connection)
    connection.commit()


def main(args: argparse.Namespace):
    if args.profile:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profile_dir

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

    if llm_args.rpd:
        recreate_trace(llm_args)

    llm = LLM(**dataclasses.asdict(llm_args.engine_args))

    prompt_param = TokensPrompt(
        prompt_token_ids=llm_args.prompt) if isinstance(
            llm_args.prompt, list) else llm_args.prompt

    if llm_args.image_path is not None:
        image = Image.open(llm_args.image_path).convert("RGB")
        prompt_param = {
            "prompt": "<|image|><|begin_of_text|>" + llm_args.prompt,
            "multi_modal_data": {
                "image": image
            }
        }

    num_tokens = 0
    start_time = time.perf_counter()
    outs = []
    with rpd_profiler_context() if args.rpd else nullcontext():
        if llm_args.serverlike:
            reqs = 0
            llm._add_request(prompt_param, llm_args.sampling_params)
            while llm.llm_engine.has_unfinished_requests():
                step_outputs = llm.llm_engine.step()
                if reqs < batch_size:
                    llm._add_request(prompt_param, llm_args.sampling_params)
                    reqs += 1
                for step_output in step_outputs:
                    if step_output.finished:
                        text = step_output.outputs[0].text
                        num_tokens += len(step_output.outputs[0].token_ids)
                        if text:
                            print(text)
        else:
            if args.profile:
                llm.start_profile()
            outs = llm.generate([prompt_param] * batch_size,
                                sampling_params=llm_args.sampling_params)
            if args.profile:
                llm.stop_profile()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if not llm_args.serverlike:
        out_lengths = [len(x.token_ids) for out in outs for x in out.outputs]
        num_tokens = sum(out_lengths)
        for out in outs:
            print("===========")
            text = out.outputs[0].text.replace('\n', ' ')
            print(f"Generated: {text}")
            if args.output_json:
                import json
                with open(args.output_json, 'w') as f:
                    json.dump({'generated': text}, f)

    if args.extra_stats:
        print(
            f"{num_tokens} tokens. {num_tokens / batch_size} on average. {num_tokens / elapsed_time:.2f} tokens/s. {elapsed_time} seconds"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='LLM Test much')
    parser.add_argument('-i',
                        '--interactive',
                        action='store_true',
                        help='Interactive mode')
    parser.add_argument('--prompt',
                        type=str,
                        default="There is a round table in the middle of the")
    parser.add_argument('--input-len', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--rpd', action='store_true')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--ignore-eos', action='store_true')
    parser.add_argument('--rpd-path', type=str, default=None)
    parser.add_argument('--image-path', type=str, default=None)
    parser.add_argument('--serverlike', action='store_true')
    parser.add_argument('--extra-stats', action='store_true')
    parser.add_argument('--output-json', type=str, default=None)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-dir', type=str, default='./vllm_profile')

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    main(args)
