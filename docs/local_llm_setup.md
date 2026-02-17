# Local LLM Setup Guide

Run SpatialAgent with local models. Three options:

1. **vLLM (NVIDIA GPUs)** - Recommended for multi-GPU setups, handles agentic workflows correctly
2. **Ollama (Cross-platform)** - Simple setup, works on macOS/Linux/Windows
3. **MLX (Apple Silicon)** - Maximum performance on Mac with Metal optimization

> **Important:** vLLM is recommended over Ollama for agentic workflows. Ollama has a [role collation issue](https://github.com/ollama/ollama/issues/5775) that causes empty responses with consecutive assistant messages, breaking the agent's tool execution loop.

---

## Option 1: vLLM (NVIDIA GPUs - Recommended)

### Architecture (vLLM)

```
┌─────────────────────────────────────────────┐
│           vLLM Server (:8000)               │
│  (OpenAI-compatible API, tensor parallel)   │
│                                             │
│  /v1/chat/completions  /v1/models           │
└─────────────────────────────────────────────┘
```

### Prerequisites (vLLM)

- Linux with NVIDIA GPU(s) (CUDA 12.x)
- Python 3.10-3.12
- 24GB+ VRAM (single GPU) or 48GB+ (dual GPU for 30B models)

### Setup (vLLM)

```bash
# Create a dedicated UV venv for vLLM (keeps dependencies isolated)
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python vllm ninja

# Verify installation
.venv-vllm/bin/python -c "import vllm; print(f'vLLM {vllm.__version__}')"
```

**Important:** FlashInfer (attention backend) uses JIT compilation and requires:
- `ninja` - installed via pip above
- `gcc` - system C compiler (install via `pacman -S gcc` on Arch, `apt install build-essential` on Ubuntu)

When running vLLM, activate the venv so ninja is found:
```bash
source .venv-vllm/bin/activate
```

### Starting the Server (vLLM)

#### Qwen3-VL Models

```bash
# Activate the vLLM venv
source .venv-vllm/bin/activate

# Dual GPU - Qwen3-VL-30B-A3B AWQ quantized (fits in 2x24GB, 32K context)
vllm serve QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 32768

```

#### Embedding Model (Qwen3-Embedding-0.6B)

The embedding model converts text to vector representations for semantic search and RAG. It runs on a separate port (8001) from the chat model (8000).

**Model details:**
- **Size:** ~1.2GB (FP16 weights)
- **Max tokens:** 8192 input, but we use 512 for efficiency
- **Output:** 1024-dimensional vectors

**Key configuration parameters:**
- `--port 8001`: Separate port from chat model
- `--max-model-len 512`: Short context for embeddings (faster, less memory)
- `--enforce-eager`: Disables CUDA graphs (reduces memory overhead for small model)
- `--gpu-memory-utilization`: Controls VRAM reservation (set low when co-located with chat model)

#### Embedding Model: GPU Configurations

**Standalone (dedicated GPU or separate from chat):**
```bash
source .venv-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.10 \
    --max-model-len 512 \
    --enforce-eager
```

**Co-located with chat model (same GPUs):**

When running embedding alongside a chat model, start embedding first with low GPU utilization, then start chat with the remaining capacity.

| GPU Setup | Embedding `-tp` | Embedding `--gpu-memory-utilization` | Chat `--gpu-memory-utilization` |
|-----------|-----------------|--------------------------------------|--------------------------------|
| 2x RTX 3090/4090 (24GB) | 2 | 0.04 (~1GB/GPU) | 0.90 (~21GB/GPU) |
| 2x L40S (48GB) | 1 | 0.03 (~1.5GB) | 0.95 (~45GB/GPU) |
| Single A100-40GB | 1 | 0.03 (~1.2GB) | 0.92 (~37GB) |
| Single A100-80GB | 1 | 0.02 (~1.6GB) | 0.95 (~76GB) |

**2x RTX 3090/4090 (24GB each):**
```bash
# Terminal 1: Embedding (split across both GPUs for lower per-GPU memory)
source .venv-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.04 \
    --max-model-len 512 \
    --enforce-eager

# Terminal 2: Chat model with remaining capacity
source .venv-vllm/bin/activate
vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64
```

**2x L40S (48GB each) or Single A100:**
```bash
# Terminal 1: Embedding on single GPU (plenty of headroom)
source .venv-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.03 \
    --max-model-len 512 \
    --enforce-eager

# Terminal 2: Chat model (adjust -tp based on your setup)
# For 2x L40S: use -tp 2 for full precision, or -tp 1 for AWQ on single GPU
# For A100-80GB: use -tp 1 with full precision
# For A100-40GB: use -tp 1 with AWQ
```

**Why split embedding across 2 GPUs on consumer cards?**
Using `-tp 2` for embedding on 2x24GB GPUs means each GPU only reserves 0.04 × 24GB ≈ 1GB for embedding, leaving more room for the chat model's KV cache. On larger GPUs (L40S, A100), this isn't necessary.

#### Ministral-3 Models

**AWQ Quantized (Recommended):**

```bash
# Dual GPU - Ministral-3-14B AWQ (128K context, CUDA graphs enabled)
source .venv-vllm/bin/activate
vllm serve cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16
```

**Full Precision (requires more VRAM):**

```bash
# Dual GPU - Ministral-3-14B full precision (32K context max due to VRAM)
source .venv-vllm/bin/activate
vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 32768
```

**Context Length Notes:**
- AWQ + FP8 KV cache: 128K with CUDA graphs, 256K with `--enforce-eager`
- Full precision: 32K max on 2x24GB due to memory constraints
- CUDA graphs improve inference speed but use more memory during warmup

### Configure Context Length (vLLM)

Use `--max-model-len` to control context window size:

| Setting | Context | VRAM Impact | Use Case |
|---------|---------|-------------|----------|
| `--max-model-len 8192` | 8K | Minimal | Quick tasks |
| `--max-model-len 32768` | 32K | Moderate | **Recommended** |
| `--max-model-len 65536` | 64K | High | Long documents |
| `--max-model-len 131072` | 128K | Very high | Maximum context |

**Note:** Larger context requires more VRAM for KV cache. If you get OOM errors, reduce `--max-model-len`.

### Output Token Limits (max_tokens)

When calling the model API, `max_tokens` controls the maximum generation length. Qwen recommends different limits based on task complexity:

| Setting | Use Case | Notes |
|---------|----------|-------|
| `max_tokens=4096` | OpenAI/Claude/Gemini models | Default for cloud APIs |
| `max_tokens=32768` | Qwen standard tasks | Recommended for most queries |
| `max_tokens=81920` | Qwen complex reasoning | For math/coding competitions |

**Source:** [Qwen3 Blog - Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)

> "We recommend using an output length of 32,768 tokens for most queries. For benchmarking on highly complex problems, such as those found in math and programming competitions, we suggest setting the max output length to 38,912 tokens."

**Important:** Using too small a `max_tokens` (e.g., 2048) can cause responses to be truncated before the model outputs its final answer, significantly hurting benchmark accuracy.

### Recommended Inference Parameters

Based on benchmark testing on HLE Biology and GPQA Diamond:

| Model | Temperature | top_p | top_k | Notes |
|-------|-------------|-------|-------|-------|
| **Qwen3-VL-32B** | 1.0 | 1.0 | -1 | vLLM defaults, best balance |
| **Ministral-3-14B** | 0.15 | 1.0 | -1 | Per Mistral docs (production) |

**Latency Comparison (linear seconds/question, 8x parallel estimate):**

| Dataset | Qwen3-VL-32B | Ministral-3-14B | Speedup |
|---------|--------------|-----------------|---------|
| HLE Biology | ~38s | ~18s | 2.1x faster |
| GPQA Diamond | ~70s | ~32s | 2.2x faster |

Ministral-3-14B (14B params) is ~2x faster than Qwen3-VL-32B (32B params) with slightly lower accuracy.

### Tensor Parallelism (Multi-GPU)

The `--tensor-parallel-size` (or `-tp`) setting controls how model weights are split across GPUs:

| Setting | Description | Use Case |
|---------|-------------|----------|
| `-tp 1` | Single GPU | Model fits in one GPU's VRAM |
| `-tp 2` | Split across 2 GPUs | Model too large for single GPU, or want more KV cache |
| `-tp 4` | Split across 4 GPUs | Very large models or maximum context |

**How it works:** Tensor parallelism shards model layers across GPUs. Each GPU holds 1/N of the weights and computes 1/N of each layer, then GPUs synchronize via NVLink/PCIe. This reduces per-GPU memory but adds communication overhead.

**When to use multi-GPU:**
- Model weights exceed single GPU VRAM
- Need more KV cache for longer context (KV cache is also split across GPUs)
- Want to serve more concurrent requests (`--max-num-seqs`)

#### GPU-Specific Configurations

**2x RTX 3090/4090 (24GB each, 48GB total):**
```bash
# Qwen3-VL-32B AWQ (~18GB weights, leaves room for 128K KV cache)
vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64

# Ministral-3-14B AWQ (~5GB weights, lots of room for KV cache)
vllm serve cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16
```

**2x L40S (48GB each, 96GB total):**
```bash
# Qwen3-VL-32B full precision (no quantization needed)
vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64

# Or run on single GPU with AWQ quantization
vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90
```

**Single A100 (40GB or 80GB):**
```bash
# A100-40GB: Use AWQ quantization
vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90

# A100-80GB: Full precision, maximum context
vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64
```

**Memory breakdown (approximate):**
| Component | Qwen3-VL-32B FP16 | Qwen3-VL-32B AWQ | Ministral-3-14B AWQ |
|-----------|-------------------|------------------|---------------------|
| Model weights | ~64GB | ~18GB | ~5GB |
| KV cache (128K) | ~16GB | ~8GB (FP8) | ~8GB (FP8) |
| Overhead | ~2-4GB | ~2-4GB | ~2-4GB |

### Available Models (vLLM)

**Chat/Vision Models:**
| Model | Vision | Min VRAM | Weights | Notes |
|-------|--------|----------|---------|-------|
| Qwen3-VL-32B FP16 | Yes | 80GB (1×A100) or 2×48GB | ~64GB | Full precision, best quality |
| Qwen3-VL-32B AWQ | Yes | 40GB (1×A100) or 2×24GB | ~18GB | AWQ 4-bit, use FP8 KV for 128K |
| Qwen3-VL-30B AWQ | Yes | 2×24GB | ~18GB | AWQ 4-bit, MoE architecture |
| Ministral-3-14B AWQ | Yes | 24GB (1×) or 2×24GB | ~5GB | AWQ 4-bit, 128K with FP8 KV |
| Ministral-3-14B FP16 | Yes | 2×24GB | ~28GB | Full precision, needs tokenizer flags |

**Embedding Model:**
| Model | VRAM | Weights | Config | Notes |
|-------|------|---------|--------|-------|
| Qwen3-Embedding-0.6B | ~1.5GB | ~1.2GB | `-tp 1`, `--enforce-eager` | 1024-dim vectors, port 8001 |

The embedding model is small enough to run on any GPU. When co-located with a chat model, use low `--gpu-memory-utilization` (0.02-0.04) to leave room for the chat model.

### LiteLLM Proxy for vLLM (Recommended for Ministral)

Using LiteLLM as a proxy in front of vLLM enables automatic handling of Ministral's `continue_final_message` requirement without setting environment variables.

#### Architecture (vLLM + LiteLLM)

```
┌─────────────────────────────────────────────────────────────┐
│                  LiteLLM Proxy (:8080)                      │
│  - Routes requests to vLLM backends                         │
│  - Custom callback sets continue_final_message dynamically  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  vLLM Chat Server (:8000)│    │ vLLM Embed Server (:8001)│
│  (Qwen3-VL-32B, etc)     │    │  (Qwen3-Embedding-0.6B)  │
└──────────────────────────┘    └──────────────────────────┘
```

#### Setup LiteLLM for vLLM

```bash
# Create separate venv for LiteLLM (keeps dependencies isolated)
uv venv .venv-litellm --python 3.12
uv pip install --python .venv-litellm/bin/python litellm

# The config and callback files are already in the repo:
# - vllm_litellm_config.yaml  (LiteLLM configuration)
# - custom_callbacks.py       (Conditional continue_final_message handler)
```

#### Start the Servers

> **Note:** The examples below are for 2×24GB GPUs (RTX 3090/4090). For other configurations:
> - **2×L40S or A100-80GB:** Use `-tp 1` for embedding, `-tp 2` (L40S) or `-tp 1` (A100) for chat with full precision
> - **A100-40GB:** Use `-tp 1` for both, AWQ quantization for chat
> - See [Embedding Model: GPU Configurations](#embedding-model-gpu-configurations) and [Tensor Parallelism](#tensor-parallelism-multi-gpu) for details

```bash
# Terminal 1: Start vLLM embedding server first (adjust -tp and GPU util for your setup)
source .venv-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.04 \
    --max-model-len 512 \
    --enforce-eager

# Terminal 2: Start vLLM chat server (choose one, adjust -tp for your setup)

# Option A: Qwen3-VL-32B AWQ (recommended - 128K context, FP8 KV cache)
source .venv-vllm/bin/activate
vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64

# Option B: Qwen3-VL-30B AWQ (MoE architecture)
source .venv-vllm/bin/activate
vllm serve QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90

# Option C: Ministral-3-14B AWQ (128K context, requires LiteLLM callback)
source .venv-vllm/bin/activate
vllm serve cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16

# Terminal 3: Start LiteLLM proxy
source .venv-litellm/bin/activate
litellm --config vllm_litellm_config.yaml --port 8080
```

#### Configuration Files

**`vllm_litellm_config.yaml`**:

```yaml
model_list:
  # Chat models via vLLM (port 8000) - use whichever is running
  - model_name: qwen3-vl-32b
    litellm_params:
      model: hosted_vllm/QuantTrio/Qwen3-VL-32B-Instruct-AWQ
      api_base: http://localhost:8000/v1
      api_key: EMPTY

  - model_name: qwen3-vl-30b
    litellm_params:
      model: hosted_vllm/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ
      api_base: http://localhost:8000/v1
      api_key: EMPTY

  - model_name: ministral-3
    litellm_params:
      model: hosted_vllm/cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit
      api_base: http://localhost:8000/v1
      api_key: EMPTY
      merge_consecutive_messages: true

  # Embeddings via vLLM (port 8001)
  - model_name: qwen3-embedding:0.6b
    litellm_params:
      model: hosted_vllm/Qwen/Qwen3-Embedding-0.6B
      api_base: http://localhost:8001/v1
      api_key: EMPTY

litellm_settings:
  callbacks: custom_callbacks.proxy_handler_instance
  drop_params: true
  modify_params: true
  num_retries: 3
  request_timeout: 600

general_settings:
  disable_spend_logs: true
  health_check_mode: off
```

**`custom_callbacks.py`** (handles Mistral's continue_final_message):

```python
from litellm.integrations.custom_logger import CustomLogger

class ContinueFinalMessageHandler(CustomLogger):
    """Conditionally set continue_final_message for Mistral models on vLLM."""

    MISTRAL_MODELS = ("mistral", "ministral", "codestral", "pixtral")

    def _is_mistral_model(self, model: str) -> bool:
        return any(name in model.lower() for name in self.MISTRAL_MODELS)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        model = data.get("model", "")
        messages = data.get("messages", [])

        # Only apply to Mistral models (Qwen handles this natively)
        if not self._is_mistral_model(model):
            return data

        # Only set continue_final_message when last message is from assistant
        if messages and messages[-1].get("role") == "assistant":
            extra_body = data.get("extra_body", {})
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
            data["extra_body"] = extra_body

        return data

proxy_handler_instance = ContinueFinalMessageHandler()
```

> **Note:** Qwen models handle consecutive assistant messages natively and don't need this callback. It only activates for Mistral family models.

#### Running the Agent (via LiteLLM)

```bash
conda activate spatialagent
export CUSTOM_MODEL_BASE_URL=http://localhost:8080/v1
export CUSTOM_MODEL_API_KEY=EMPTY
export CUSTOM_EMBED_BASE_URL=http://localhost:8080/v1
export CUSTOM_EMBED_API_KEY=EMPTY
export CUSTOM_EMBED_MODEL=qwen3-embedding:0.6b
export TOKENIZERS_PARALLELISM=false
```

```python
from spatialagent.agent import SpatialAgent, make_llm

# Qwen3-VL-32B recommended (handles consecutive messages natively)
llm = make_llm("qwen3-vl-32b", temperature=0)
agent = SpatialAgent(llm=llm, save_path="./experiments/local/")

result = agent.run(
    "What tools do you have available?",
    config={"thread_id": "test_litellm"}
)
```

#### Parallel Benchmarking

Use the included script to run all benchmark problems in parallel:

```bash
chmod +x run_benchmark_parallel.sh
./run_benchmark_parallel.sh qwen3-vl-32b
```

This runs all 5 test problems simultaneously, with logs saved to `experiments/parallel_benchmark_<model>_<timestamp>/`.

---

## Option 2: Ollama (Cross-platform)

> **Warning:** Ollama has a [role collation issue](https://github.com/ollama/ollama/issues/5775) that merges consecutive assistant messages, causing the agent to return empty responses after tool execution. Use vLLM for production agentic workflows.

### Architecture (Ollama)

```
┌─────────────────────────────────────────────┐
│           Ollama Server (:11434)            │
│  (chat, vision, embeddings - all-in-one)    │
│                                             │
│  /v1/chat/completions  /v1/embeddings       │
└─────────────────────────────────────────────┘
```

### Prerequisites (Ollama)

- Any Mac, Linux, or Windows machine
- [Ollama](https://ollama.com/download) installed

### Setup (Ollama)

```bash
# Install Ollama (macOS)
brew install ollama

# Start the server
ollama serve

# Pull required models (in another terminal)
ollama pull qwen3-vl:30b          # Vision model, MoE (~20GB) - recommended
ollama pull qwen3-vl:32b          # Vision model, dense (~21GB)
ollama pull ministral-3:14b       # Fast text model (~9GB)
ollama pull qwen3-embedding:0.6b  # Embeddings (~639MB, recommended)
# Alternative embedding models:
# ollama pull nomic-embed-text-v2-moe  # (~957MB)
# ollama pull embeddinggemma:300m      # (~621MB)
```

### Configure Context Length (128K)

Ollama defaults to a small context window (~2K). Set 128K context globally:

```bash
# Set default context to 128K (environment variable)
export OLLAMA_CONTEXT_LENGTH=131072

# For systemd service, add to /etc/systemd/system/ollama.service:
# Environment="OLLAMA_CONTEXT_LENGTH=131072"
```

Restart `ollama serve` after setting this. For 256K context, use `262144` instead.

**Note:** Larger context requires more RAM. 128K context adds ~8-16GB RAM usage depending on the model.

### Running the Agent (Ollama)

```bash
conda activate spatialagent
export CUSTOM_MODEL_BASE_URL=http://localhost:11434/v1
export CUSTOM_EMBED_BASE_URL=http://localhost:11434/v1
export CUSTOM_EMBED_MODEL=qwen3-embedding:0.6b
export TOKENIZERS_PARALLELISM=false  # Suppress tokenizer warnings
```

Then in Python:

```python
from spatialagent.agent import SpatialAgent, make_llm

llm = make_llm("qwen3-vl:30b")  # recommended, or ministral-3:14b, qwen3-vl:32b
agent = SpatialAgent(llm=llm, save_path="./experiments/local/")

result = agent.run(
    "What tools do you have available?",
    config={"thread_id": "test_ollama"}
)
```

### Available Models (Ollama)

| Alias | Model | Vision | Max Context | Size | Notes |
|-------|-------|--------|-------------|------|-------|
| `qwen3-vl:30b` | Qwen3-VL-30B-A3B | Yes | 256K | ~20GB | MoE architecture |
| `qwen3-vl:32b` | Qwen3-VL-32B | Yes | 128K | ~21GB | Dense model |
| `ministral-3:14b` | Ministral-3-14B | Yes | 256K | ~9GB | Dense model |
| `qwen3-embedding:0.6b` | Qwen3-Embedding 0.6B | - | - | ~639MB | Embeddings (recommended) |
| `nomic-embed-text-v2-moe` | Nomic Embed v2 MoE | - | - | ~957MB | Embeddings |
| `embeddinggemma:300m` | EmbeddingGemma 300M | - | - | ~621MB | Embeddings |

---

## Option 3: MLX (Apple Silicon)

### Architecture (MLX)

```
┌───────────────────────────────────────────────────────────────────┐
│                      LiteLLM Proxy (:8080)                        │
│                 (routing, message normalization)                  │
└────────┬─────────────────────┬─────────────────────┬──────────────┘
         │                     │                     │
┌────────▼────────┐  ┌─────────▼─────────┐  ┌───────▼────────┐
│  mlx_vlm.server │  │  local_mlx_lm     │  │ local_embed    │
│  (:8081)        │  │  (:8083)          │  │  (:8082)       │
│  Vision models  │  │  Text-only models │  │  Embeddings    │
└─────────────────┘  └───────────────────┘  └────────────────┘
```

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12
- UV package manager

## Setup

### 1. Create the LLM Server Environment

```bash
uv venv .venv-llm --python 3.12
source .venv-llm/bin/activate
uv pip install -r requirements-llm.txt
```

### 2. Create the Agent Environment

```bash
conda create -n spatialagent python=3.12 -y
conda activate spatialagent
pip install -r requirements.txt
pip install -e external/utag
```

## Running the Servers

### Quick Start (Recommended)

```bash
# Terminal 1: Start all LLM servers
./start_local_servers.sh

# Terminal 2: Run the agent
conda activate spatialagent
export CUSTOM_MODEL_BASE_URL=http://localhost:8080/v1
export CUSTOM_EMBED_BASE_URL=http://localhost:8080/v1
export CUSTOM_EMBED_MODEL=qwen  # or nomic, nomic-v2, qwen-small
export TOKENIZERS_PARALLELISM=false
python -c "
from spatialagent.agent import SpatialAgent, make_llm
llm = make_llm('qwen3-vl-30b-a3b')
agent = SpatialAgent(llm=llm, save_path='./experiments/local/')
result = agent.run('What tools do you have available?', config={'thread_id': 'test'})
"

# When done
./start_local_servers.sh stop
```

Logs are written to `logs/` directory. Use `./start_local_servers.sh status` to check server status.

### Manual Start (Individual Terminals)

### Terminal 1: MLX VLM Server (Vision models)

```bash
source .venv-llm/bin/activate
python -m mlx_vlm.server --port 8081
```

### Terminal 2: MLX LM Server (Text-only models)

```bash
source .venv-llm/bin/activate
python local_mlx_lm_server.py --port 8083
```

### Terminal 3: Embeddings Server

```bash
source .venv-llm/bin/activate
python local_embed_server.py --model qwen --port 8082
# Alternative models: nomic, nomic-v2, qwen-small
```

### Terminal 4: LiteLLM Proxy

```bash
source .venv-llm/bin/activate
litellm --config local_litellm_config.yaml --port 8080
```

## Running the Agent

```bash
conda activate spatialagent
export CUSTOM_MODEL_BASE_URL=http://localhost:8080/v1
export CUSTOM_EMBED_BASE_URL=http://localhost:8080/v1
export CUSTOM_EMBED_MODEL=qwen  # or nomic, nomic-v2, qwen-small
export TOKENIZERS_PARALLELISM=false
```

Then in Python:

```python
from spatialagent.agent import SpatialAgent, make_llm

llm = make_llm("qwen3-vl-30b-a3b")  # recommended, or ministral-14b, qwen3-vl-32b
agent = SpatialAgent(llm=llm, save_path="./experiments/local/")

result = agent.run(
    "What tools do you have available?",
    config={"thread_id": "test_1"}
)
```

## Available Models

| Alias | Model | Vision | Server | Notes |
|-------|-------|--------|--------|-------|
| `qwen3-vl-30b-a3b` | Qwen3-VL-30B-A3B-Instruct-8bit | Yes | mlx_vlm | MoE architecture |
| `qwen3-vl-32b` | Qwen3-VL-32B-Instruct-8bit | Yes | mlx_vlm | Dense model |
| `ministral-14b` | Ministral-3-14B-Instruct-2512-8bit | No | mlx_lm | mlx_vlm not yet supported |

## Benchmark Results (M4 MacBook Pro 128GB)

| Model | RAM Delta | Total RAM | Time | Notes |
|-------|-----------|-----------|------|-------|
| ministral-14b | +14.1 GB | 39.3 GB | 3.5s | Not supported in mlx_vlm yet |
| qwen3-vl-30b-a3b | +31.3 GB | 57.1 GB | 7.4s | Vision + MoE |
| qwen3-vl-32b | +33.4 GB | 59.2 GB | 10.4s | Vision, dense model |

All models unload cleanly back to ~25 GB baseline when idle.

## Configuration

The `local_litellm_config.yaml` routes requests to the appropriate backend servers:

```yaml
model_list:
  # Vision models via mlx_vlm (port 8081)
  - model_name: qwen3-vl-32b
    litellm_params:
      model: hosted_vllm/mlx-community/Qwen3-VL-32B-Instruct-8bit
      api_base: http://localhost:8081
      api_key: fake-key
    model_info:
      health_check_model: skip

  # Text-only models via mlx_lm (port 8083)
  - model_name: ministral-14b
    litellm_params:
      model: hosted_vllm/mlx-community/Ministral-3-14B-Instruct-2512-8bit
      api_base: http://localhost:8083
      api_key: fake-key
    model_info:
      health_check_model: skip

  # Embeddings via local_embed_server (port 8082)
  - model_name: qwen
    litellm_params:
      model: openai/qwen
      api_base: http://localhost:8082/v1
      api_key: fake-key
    model_info:
      mode: embedding

litellm_settings:
  drop_params: true
  modify_params: true
  num_retries: 3
  request_timeout: 300
```

**Notes**:
- Use `hosted_vllm/` prefix for MLX servers (avoids `/v1` path auto-append)
- `health_check_model: skip` prevents LiteLLM from loading models on startup

---

## Why LiteLLM Proxy with vLLM?

You can use vLLM directly for simple inference, but LiteLLM proxy adds important features for agentic workflows:

| Feature | vLLM Direct | vLLM + LiteLLM Proxy |
|---------|-------------|---------------------|
| **Model serving** | One model per port | Route to multiple models/ports |
| **Message handling** | Raw passthrough | `merge_consecutive_messages` support |
| **Ministral support** | ❌ Requires manual `continue_final_message` | ✅ Custom callback handles it automatically |
| **Multiple backends** | Single vLLM server | Route to vLLM, Ollama, cloud APIs |
| **Complexity** | Simple | Adds one more service |

### When to use LiteLLM Proxy

**Use LiteLLM if:**
- Running **Ministral models** - they require `continue_final_message=true` when the last message is from assistant (agentic workflows)
- Need to route between **multiple models** on different ports
- Want a **unified endpoint** for both chat and embedding models

**Skip LiteLLM if:**
- Using **Qwen models only** - they handle consecutive messages natively
- Simple single-model deployment
- Minimizing infrastructure complexity

### How the Ministral callback works

In agentic workflows, the agent often generates partial responses that end with an assistant message (e.g., tool calls). Ministral requires `continue_final_message=true` to continue from that point. The custom callback (`custom_callbacks.py`) detects this automatically:

```python
# If last message is from assistant, set continue_final_message
if messages and messages[-1].get("role") == "assistant":
    extra_body["continue_final_message"] = True
```

---

## Comparison: vLLM vs Ollama vs MLX

| Feature | vLLM | Ollama | MLX |
|---------|------|--------|-----|
| Platform | Linux (NVIDIA) | macOS, Linux, Windows | Apple Silicon only |
| Multi-GPU | Yes (tensor parallel) | No | No |
| Agentic workflows | ✅ Works correctly | ⚠️ Role collation issue | ✅ Works correctly |
| `continue_final_message` | ✅ Supported | ❌ Ignored | N/A |
| Context length | `--max-model-len` | `OLLAMA_CONTEXT_LENGTH` | Per-model config |
| Setup complexity | `pip install vllm` | `brew install ollama` | Multiple servers + LiteLLM |
| Model management | HuggingFace auto-download | `ollama pull` | Manual download |
| Quantization | AWQ, GPTQ, FP8 | GGUF (various) | 8-bit MLX format |
| Best for | Production NVIDIA | Easy cross-platform | Max Mac performance |

### Agentic Workflow Compatibility

| Backend | Qwen3-VL | Ministral-3 | Notes |
|---------|----------|-------------|-------|
| **vLLM + LiteLLM** | ✅ Works | ✅ Works | **Recommended** - callback handles continue_final_message |
| **Ollama** | ❌ Empty response | ❌ Empty response | Role collation issue |
| **MLX** | ✅ Works | ✅ Works | Apple Silicon only |

---

## Troubleshooting

### vLLM Issues

**CUDA out of memory?** Reduce context length with `--max-model-len 16384`, use quantized models (AWQ), or increase `--tensor-parallel-size`.

**Model not found?** Use exact HuggingFace model path (e.g., `Qwen/Qwen3-VL-8B-Instruct`).

**Slow first response?** vLLM compiles CUDA graphs on first request. Subsequent requests are faster.

**Ministral "add_generation_prompt" error?** Use LiteLLM proxy with the custom callback (see LiteLLM section above).

**FP8 warning on RTX 3090/4090?** Normal - vLLM uses Marlin kernel for weight-only FP8 on non-Hopper GPUs. Performance is still good.

**Check server status:** `curl http://localhost:8000/v1/models` lists loaded models.

**Check GPU memory:** `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`

### MLX Issues

**Model download slow?** Models download from Hugging Face on first use (~5-10GB each).

**Out of memory?** Use `qwen3-vl-30b` (MoE, uses less memory) instead of `qwen3-vl-32b`.

**Tokenizer errors?** Ensure `transformers==5.0.0rc1` (not dev versions).

**Port in use?** Check with `lsof -i :8080` and kill conflicting processes.

### LiteLLM Issues

**Callback not loading?** Ensure you're running LiteLLM from the project directory so it can find `custom_callbacks.py`.

**404 errors?** Check that vLLM is running and the model name in config matches what vLLM is serving: `curl http://localhost:8000/v1/models`.

**"Cannot set continue_final_message when last message is not from assistant"?** This means the callback isn't working. Verify callback is loaded in LiteLLM startup logs.

**Embedding errors?** Ensure Ollama is running and has the embedding model: `ollama pull qwen3-embedding:0.6b`.

**Check LiteLLM status:** `curl http://localhost:8080/v1/models` lists configured models.

### Ollama Issues

**Model not found?** Run `ollama list` to see downloaded models. Pull with `ollama pull <model>`.

**Slow first response?** Ollama loads models on first request. Subsequent requests are faster.

**VRAM issues?** Use smaller quantizations: `ollama pull qwen3-vl:30b-q4_0` instead of default.

**Empty responses after tool execution?** This is the [role collation issue](https://github.com/ollama/ollama/issues/5775). Switch to vLLM for agentic workflows.

**Check server status:** `curl http://localhost:11434/api/tags` lists available models.
