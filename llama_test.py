from vllm import LLM, SamplingParams
import time
from PIL import Image

def main():
    llm = LLM(
        #'/models/Llama-3.2-11B-Vision-Instruct-FP8-KV',
        '/models/Llama-3.2-90B-Vision-Instruct',
        tensor_parallel_size=4,
        #quantization="fp8",
        #enforce_eager=True,
        #kv_cache_dtype="fp8",
    )
    batch_size = 1
    max_tokens = 256
    prompt = f"<|image|><|begin_of_text|>Describe image in two sentences"

    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     max_tokens=max_tokens)

    start_time = time.perf_counter()
    image = Image.open("/projects/image1.jpg") \
            .convert("RGB")

    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}},

    outs = llm.generate(inputs, sampling_params=sampling_params)
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
    main()
