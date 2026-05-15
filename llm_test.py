import argparse
import dataclasses
import json
import os
import time

import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm import LLM, SamplingParams
from vllm.inputs.data import PromptType, TokensPrompt
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
from PIL import Image
try:
    from vllm.utils.async_utils import merge_async_iterators
except ModuleNotFoundError:
    try:
        from vllm.utils.asyncio import merge_async_iterators
    except ModuleNotFoundError:
        from vllm.utils import merge_async_iterators


class LlmKwargs(dict):

    def __init__(self, args: argparse.Namespace) -> None:
        self.engine_args = EngineArgs.from_cli_args(args)
        self.async_engine_args = AsyncEngineArgs.from_cli_args(args)
        self.prompt = args.prompt if args.input_len is None else [
            0
        ] * args.input_len
        self.batch_size = args.batch_size
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
        )
        self.image_path = args.image_path
        self.serverlike = args.serverlike
        self.async_engine = args.async_engine
        self.dataset_path = args.dataset_path
        self.input_len = args.input_len

    def __setitem__(self, key: str, value: str) -> None:
        self.engine_args = dataclasses.replace(self.engine_args,
                                               **{key: value})
        self.async_engine_args = dataclasses.replace(self.async_engine_args,
                                                     **{key: value})

    def __str__(self) -> str:
        res = "\n === Engine Args === \n"
        res += f"{self.async_engine_args if self.async_engine else self.engine_args}\n"
        res += "\n === Sampling params ===\n"
        res += f"{self.sampling_params}\n"
        res += "\n === Misc === \n"
        res += f"Prompt: {self.prompt}\n"
        res += f"Batch size: {self.batch_size}\n"
        res += f"Image path: {self.image_path}\n"
        res += f"Serverlike: {self.serverlike}\n"
        res += f"Async engine: {self.async_engine}\n"
        res += f"Dataset path: {self.dataset_path}\n"
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
    llm_kwargs.input_len = int(input("Enter input length: "))
    llm_kwargs.prompt = [0] * llm_kwargs.input_len


def select_temperature(llm_kwargs: LlmKwargs):
    llm_kwargs.sampling_params.temperature = float(
        input("Enter temperature: "))


def select_ignore_eos(llm_kwargs: LlmKwargs):
    llm_kwargs.sampling_params.ignore_eos = [False, True][menu([False, True])]


def select_image_path(llm_kwargs: LlmKwargs):
    llm_kwargs.image_path = input("Enter image path: ")


def select_serverlike(llm_kwargs: LlmKwargs):
    llm_kwargs.serverlike = [False, True][menu([False, True])]


def select_async_engine(llm_kwargs: LlmKwargs):
    llm_kwargs.async_engine = [False, True][menu([False, True])]


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
    "image_path": select_image_path,
    "serverlike": select_serverlike,
    "async_engine": select_async_engine,
    "Done": None,
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
            if new_value == "None":
                new_value = None
            llm_kwargs[key] = new_value
        print(llm_kwargs)


async def run_async(
    engine_args: AsyncEngineArgs,
    sampling_params: SamplingParams,
    prompts: PromptType,
):
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args, )

    async with build_async_engine_client_from_engine_args(engine_args,
                                                          False) as llm:
        generators = []
        for i, prompt in enumerate(prompts):
            generator = llm.generate(prompt,
                                     sampling_params,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            for output in res.outputs:
                if output.finish_reason or output.stop_reason:
                    print("===========")
                    print(f"Prompt: {res.prompt}")
                    print(f"Generated: {output.text}")


def make_prompts(llm_args: LlmKwargs):
    if llm_args.dataset_path is not None:
        with open(llm_args.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        data = [
            entry for entry in data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        prompts = []
        for entry in data:
            if len(prompts) >= llm_args.batch_size:
                break
            prompt = entry["conversations"][0]["value"]
            if llm_args.input_len and len(prompt) > llm_args.input_len:
                prompt = prompt[:llm_args.input_len]
            prompts.append(prompt)
        return prompts

    prompt_param = (TokensPrompt(prompt_token_ids=llm_args.prompt)
                    if isinstance(llm_args.prompt, list) else llm_args.prompt)
    if llm_args.image_path is not None:
        image = Image.open(llm_args.image_path).convert("RGB")
        prompt_param = {
            "prompt": "<|image|><|begin_of_text|>" + llm_args.prompt,
            "multi_modal_data": {
                "image": image
            },
        }
    return [prompt_param] * llm_args.batch_size


def main(args: argparse.Namespace):

    llm_args = LlmKwargs(args)
    if args.profile:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profile_dir
        llm_args.engine_args.profiler_config.profiler = "torch"
        llm_args.engine_args.profiler_config.torch_profiler_dir = args.profile_dir
        llm_args.engine_args.profiler_config.torch_profiler_with_stack = args.profile_with_stack

    print(llm_args)
    if args.interactive:
        interactive(llm_args)

    batch_size = llm_args.batch_size

    prompts = make_prompts(llm_args)

    num_tokens = 0
    start_time = time.perf_counter()
    outs = []

    if llm_args.async_engine:
        uvloop.run(
            run_async(llm_args.async_engine_args, llm_args.sampling_params,
                      prompts))
    elif llm_args.serverlike:
        llm = LLM(**dataclasses.asdict(llm_args.engine_args))
        reqs = 0
        llm._add_request(prompts[reqs], llm_args.sampling_params)
        while llm.llm_engine.has_unfinished_requests():
            step_outputs = llm.llm_engine.step()
            if reqs < batch_size - 1:
                reqs += 1
                llm._add_request(prompts[reqs], llm_args.sampling_params)
            for step_output in step_outputs:
                if step_output.finished:
                    text = step_output.outputs[0].text
                    num_tokens += len(step_output.outputs[0].token_ids)
                    if text:
                        print(text)
    else:
        llm = LLM(**dataclasses.asdict(llm_args.engine_args))
        if args.profile:
            llm.start_profile()
        outs = llm.generate(prompts, sampling_params=llm_args.sampling_params)
        if args.profile:
            llm.stop_profile()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if not llm_args.serverlike:
        out_lengths = [len(x.token_ids) for out in outs for x in out.outputs]
        num_tokens = sum(out_lengths)
        outputs_json = []
        for out in outs:
            print("===========")
            text = out.outputs[0].text.replace("\n", " ")
            print(f"Prompt: {out.prompt}")
            print(f"Generated: {text}")
            outputs_json.append({
                "prompt": out.prompt,
                "generated": text,
            })
        if args.output_json:
            import json

            with open(args.output_json, "w") as f:
                json.dump({"results": outputs_json}, f)

    if args.extra_stats:
        print(
            f"{num_tokens} tokens. {num_tokens / batch_size} on average. {num_tokens / elapsed_time:.2f} tokens/s. {elapsed_time} seconds"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="LLM Test much")
    parser.add_argument("-i",
                        "--interactive",
                        action="store_true",
                        help="Interactive mode")
    parser.add_argument("--prompt",
                        type=str,
                        default="There is a round table in the middle of the")
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--serverlike", action="store_true")
    parser.add_argument("--async-engine", action="store_true")
    parser.add_argument("--extra-stats", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="./vllm_profile")
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--dataset-path", type=str, default=None)

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    main(args)
