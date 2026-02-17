## SpatialAgent — Local LLM Branch

This branch runs SpatialAgent entirely with **locally-served LLMs** — no API keys required. The `web_search` tool is commented out since it depends on cloud APIs (Anthropic/OpenAI/Google).

For the main branch with full cloud API support, see the `main` branch.

### Prerequisites

- Python 3.11, Conda
- **NVIDIA GPUs** (vLLM backend) or **Apple Silicon Mac** (MLX backend)

### Step 1: Set Up the Agent Environment

```bash
./setup_env.sh              # Creates 'spatial_agent' conda environment
conda activate spatial_agent
```

### Step 2: Set Up Local LLM Servers

**For Linux + NVIDIA GPUs (vLLM):**

```bash
./local_llm/vllm/setup.sh                # One-time setup (creates .venv-vllm, .venv-litellm)

./local_llm/vllm/start.sh                # Start with Qwen3-VL-32B (default)
./local_llm/vllm/start.sh ministral      # Or start with Ministral-3-14B
./local_llm/vllm/start.sh status         # Check server status
./local_llm/vllm/start.sh stop           # Stop all servers
```

**For macOS + Apple Silicon (MLX):**

```bash
./local_llm/mlx/setup.sh                 # One-time setup (creates .venv-mlx)

./local_llm/mlx/start.sh                 # Start servers
./local_llm/mlx/start.sh status          # Check server status
./local_llm/mlx/start.sh stop            # Stop all servers
```

### Step 3: Run SpatialAgent

After starting the servers, set the environment variables and run. See `main.ipynb` for a full walkthrough.

```bash
# Point SpatialAgent to local servers
export CUSTOM_MODEL_BASE_URL=http://localhost:8088/v1   # LiteLLM proxy (vLLM)
export CUSTOM_EMBED_BASE_URL=http://localhost:8088/v1
export CUSTOM_EMBED_MODEL=qwen3-embedding
export TOKENIZERS_PARALLELISM=false
```

```python
from spatialagent.agent import SpatialAgent, make_llm

llm = make_llm("qwen3-vl-32b")
agent = SpatialAgent(llm=llm, save_path="./experiments/local/")

result = agent.run(
    "Load the MERFISH mouse liver dataset at './data/example_merfish.h5ad', "
    "run spatial clustering, and annotate cell types using PanglaoDB markers.",
    config={"thread_id": "local_demo"}
)
```

### Project Structure

```
SpatialAgent/
├── local_llm/
│   ├── vllm/              # vLLM server scripts (Linux + NVIDIA)
│   ├── mlx/               # MLX server scripts (macOS + Apple Silicon)
│   └── shared/            # Shared configs (custom callbacks)
├── spatialagent/
│   ├── agent/             # Agent implementation
│   ├── skill/             # Skill templates (17 guided workflows)
│   ├── tool/              # Tool implementations (72 tools)
│   └── hooks.py           # Event hooks
├── benchmarks/            # Local model benchmark scripts
├── evaluation/            # Evaluation modules
├── data/                  # Reference databases (CellMarker, PanglaoDB, CZI catalog)
├── docs/                  # Documentation
├── main.ipynb             # Quick start notebook
└── setup_env.sh           # Environment setup
```

### Supported Local Models

| Model | Backend | Description |
|-------|---------|-------------|
| Qwen3-VL-32B | vLLM | Vision-language model (default) |
| Ministral-3-14B | vLLM | Mistral's lightweight model |
| MLX models | MLX | Apple Silicon optimized |

See [`docs/local_llm_setup.md`](docs/local_llm_setup.md) for full configuration details.

### Citation

```bibtex
@article{spatialagent,
	author = {Hanchen Wang and Yichun He and Coelho Paula and Matthew Bucci and Abbas Nazir and other},
	title = {SpatialAgent: An autonomous AI agent for spatial biology},
	doi = {10.1101/2025.04.01.646459},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/04/01/2024.04.01.646459},
	journal = {bioRxiv},
	year = {2025},
}
```

### License

MIT License. See [LICENSE.txt](LICENSE.txt).
