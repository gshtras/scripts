# vLLM Issue Reporting Template

**Before submitting:**
1. Check if your issue is listed in [known issues](https://github.com/ROCm/vllm/issues).

## General Information (Required for All Issues)
- **vLLM Version**: (from `pip show vllm` or git commit hash)
- **Hardware Setup**:
  - GPU(s): (Make, Model, and Count)
  - Driver Version: (`nvidia-smi` or `rocm-smi` output)
  - Memory: (Host and GPU memory)
- **Execution Environment**:
  - Docker Image: (Name + Tag)
  - CUDA/ROCm Version:
  - Python Version:
  - Kernel Version: (`uname -a`)

---

## Performance Regression Report

### 1. Benchmark Command
```bash
# Full command from benchmarks/ directory
# Include all parameters and quantization flags
# Example:
python benchmarks/benchmark_throughput.py \
  --model meta-llama/Llama-2-7b-hf \
  --tensor-parallel-size 2 \
  --dtype half \
  --num-prompts 64 \
  --input-len 1024 \
  --output-len 128 \
  -tp 8
```

### 2. Environment Configuration
```bash
# Any non-default environment variables
# Example:
export VLLM_USE_TRITON_FLASH_ATTN=False
```

### 3. Performance Metrics
| Metric               | Good Performance (Image: `vllm:old`) | Regressed Performance (Image: `vllm:new`) |
|----------------------|--------------------------------------|-------------------------------------------|
| Throughput (tokens/s)| 1250                                 | 840                                       |
| Memory Utilization   | 78%                                  | 92%                                       |
| GPU Utilization      | 95%                                  | 68%                                       |

### 4. Reproducibility Context
- Original working Docker image: `docker pull rocm/vllm-dev:main`
- Regression Docker image: `docker pull rocm/vllm-dev:nightly`
- [ ] Performance difference persists across multiple runs
- [ ] Verified with different input sizes/batch sizes

---

## Crash/Bug Report

### 1. Reproduction Steps
```bash
# Minimal command that triggers the issue
# Include deployment commands if applicable
python benchmarks/benchmark_latency.py \
  --model meta-llama/Llama-2-7b-hf \
  --max-num-seqs 16 \
  --enforce-eager
```

### 2. Error Logs
<details>
<summary>Expand for full logs</summary>

```plaintext
[Full plaintext log output]
```
</details>

### 3. Environment Context
```bash
# Non-default configurations
export VLLM_USE_TRITON_FLASH_ATTN=false
```

### 4. Diagnostic Information
- [ ] Issue reproduces with `--enforce-eager` mode
- [ ] Issue reproduces with different random seeds

---

## Additional Context
1. First observed date:
2. Frequency: (Always/Intermittent/Specific Conditions)
3. Related components: (e.g., FP8 quantization, PagedAttention)
4. Custom modifications: (List any code/configuration changes)
