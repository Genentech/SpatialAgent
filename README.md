## SpatialAgent

This is the official implementation for **SpatialAgent: An autonomous AI agent for spatial biology**.

Contact wang.hanchen@gene.com or hanchenw@stanford.edu if you have any questions.

![teaser](teaser.png)

### Overview

SpatialAgent is an autonomous AI agent for spatial transcriptomics, single-cell RNA-seq, and molecular biology. It integrates large language models with dynamic tool execution and adaptive reasoning, spanning the entire research workflow from experimental design to multimodal data analysis and hypothesis generation.

Key features:
- **Plan-Act-Conclude architecture** with direct code execution
- **72 specialized tools** for database queries, literature mining, spatial analytics, and genomic analysis
- **17 skill templates** for guided workflows (annotation, CCI, panel design, spatial mapping, etc.)
- **Multi-model support**: Claude, GPT, and Gemini model families

### Installation

```bash
# Setup environment
./setup_env.sh              # Creates 'spatial_agent' environment, python 3.11
conda activate spatial_agent

# Set API keys
export ANTHROPIC_API_KEY=your_key    # For Claude models
export OPENAI_API_KEY=your_key       # For GPT models
export GOOGLE_API_KEY=your_key       # For Gemini models (optional)
```

### Quick Start

See `main.ipynb` for a quick overview.

```python
from spatialagent.agent import SpatialAgent, make_llm

llm = make_llm("claude-sonnet-4-5-20250929")
agent = SpatialAgent(llm=llm, save_path="./experiments/demo/")

result = agent.run(
    "Find mouse brain cortex datasets from CZI and analyze neuronal cell types",
    config={"thread_id": "analysis_1"}
)
```

### Project Structure

```
SpatialAgent/
├── spatialagent/
│   ├── agent/              # Agent implementation
│   ├── skill/              # Skill templates (17 guided workflows)
│   ├── tool/               # Tool implementations (72 tools)
│   └── hooks.py            # Event hooks
├── data/                   # Reference databases (CellMarker, PanglaoDB, CZI catalog)
├── resource/               # Dependencies and external packages
├── notebooks/              # Example notebooks
├── docs/                   # Documentation
├── main.ipynb              # Quick start notebook
└── setup_env.sh            # Environment setup
```

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
