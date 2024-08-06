from vllm import LLM, SamplingParams
import time

def main():
    llm = LLM(
        '/models/models--mistralai--Mixtral-8x22B-Instruct-v0.1/snapshots/1702b01b7da3a85dbf608a83595541ba22294625/',
        tensor_parallel_size=2,
        #quantization="serenity",
        dtype='float16',
        #swap_space=16,
        #enforce_eager=True,
        #kv_cache_dtype="fp8",
        #quantization="fp8",
        #quantized_weights_path="/quantized/quark/llama.safetensors",
        #worker_use_ray=True,
        #trust_remote_code=True,
        distributed_executor_backend="mp",
    )
    batch_size = 2
    max_tokens = 256
    prompt = """Her name is"""
    sampling_params = SamplingParams(temperature=0,
                                     top_p=0.95,
                                     max_tokens=max_tokens)

    start_time = time.perf_counter()
    outs = llm.generate([prompt] * batch_size, sampling_params=sampling_params)
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
