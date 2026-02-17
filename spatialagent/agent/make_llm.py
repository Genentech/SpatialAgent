"""
Simplified LLM Factory for Azure OpenAI, OpenAI, Anthropic Claude, Google Gemini, and AWS Bedrock
"""

import os
from typing import Any, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# =============================================================================
# Alternative Approach: LangChain Wrapper for Mistral continue_final_message
# =============================================================================
# Mistral models on vLLM require special handling for multi-turn conversations:
# - When last message is from assistant: continue_final_message=True
# - When last message is from user: add_generation_prompt=True (default)
#
# We use a LiteLLM callback (local_llm/vllm/custom_callbacks.py) to handle this
# at the proxy layer. The callback automatically detects the last message role
# and sets the appropriate flags before forwarding to vLLM.
#
# An alternative approach is to use a LangChain wrapper (MistralChatOpenAI)
# that intercepts invoke() calls and adds extra_body parameters dynamically.
# This is commented out below but can be enabled if not using LiteLLM.
#
# from langchain_core.messages import BaseMessage
#
# MISTRAL_MODELS = ("mistral", "ministral", "codestral", "pixtral")
#
# def _is_mistral_model(model: str) -> bool:
#     """Check if the model is a Mistral family model."""
#     return any(name in model.lower() for name in MISTRAL_MODELS)
#
# class MistralChatOpenAI:
#     """Wrapper for ChatOpenAI that handles Mistral's continue_final_message."""
#
#     def __init__(self, **kwargs):
#         from langchain_openai import ChatOpenAI
#         self._llm = ChatOpenAI(**kwargs)
#
#     def _get_extra_body(self, messages: List[BaseMessage]) -> dict:
#         if not messages:
#             return {}
#         last_role = getattr(messages[-1], "type", None)
#         if last_role in ("ai", "assistant"):
#             return {"continue_final_message": True, "add_generation_prompt": False}
#         return {}
#
#     def invoke(self, messages: Any, **kwargs) -> Any:
#         if isinstance(messages, list):
#             extra_body = self._get_extra_body(messages)
#             if extra_body:
#                 existing = kwargs.get("extra_body", {})
#                 kwargs["extra_body"] = {**existing, **extra_body}
#         return self._llm.invoke(messages, **kwargs)
#
#     async def ainvoke(self, messages: Any, **kwargs) -> Any:
#         if isinstance(messages, list):
#             extra_body = self._get_extra_body(messages)
#             if extra_body:
#                 existing = kwargs.get("extra_body", {})
#                 kwargs["extra_body"] = {**existing, **extra_body}
#         return await self._llm.ainvoke(messages, **kwargs)
#
#     def __getattr__(self, name: str) -> Any:
#         return getattr(self._llm, name)
#
# To use the wrapper approach, in make_llm() replace:
#     return ChatOpenAI(**llm_kwargs)
# with:
#     if _is_mistral_model(model):
#         return MistralChatOpenAI(**llm_kwargs)
#     return ChatOpenAI(**llm_kwargs)
# =============================================================================

# Default recommended models (2025)
DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"

# Default stop sequences - prevent multiple action blocks and observation hallucination
DEFAULT_STOP_SEQUENCES = ["</act>", "</conclude>"]

# Models that don't support stop sequences (for string matching in fallback paths)
NO_STOP_MODELS = ("gpt-5", "gpt5")

# Bedrock model prefixes (Claude via AWS)
BEDROCK_MODEL_PREFIXES = (
    "anthropic.claude-",      # Standard Bedrock Claude
    "us.anthropic.claude-",   # Cross-region inference
    "amazon.titan-",
    "meta.llama-",
    "mistral.",
    "cohere.",
    "ai21.",
)


class BedrockConfig:
    """AWS Bedrock configuration using SSO profile."""
    PROFILE_NAME = os.environ.get("AWS_PROFILE", "spatialagent")
    REGION = os.environ.get("AWS_REGION", "us-west-2")


# Extended Thinking for Bedrock (uncomment to use)
# Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-extended-thinking.html
# def bedrock_converse_with_thinking(model_id, messages, budget_tokens=10000, max_tokens=16000):
#     import boto3
#     client = boto3.Session(profile_name="spatialagent").client("bedrock-runtime", region_name="us-west-2")
#     response = client.converse(
#         modelId=model_id, messages=messages,
#         inferenceConfig={"maxTokens": max_tokens},
#         additionalModelRequestFields={"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
#     )
#     result = {"thinking": "", "text": "", "usage": response.get("usage", {})}
#     for block in response.get("output", {}).get("message", {}).get("content", []):
#         if "reasoningContent" in block:
#             result["thinking"] = block["reasoningContent"].get("reasoningText", {}).get("text", "")
#         elif "text" in block:
#             result["text"] = block["text"]
#     return result


def _is_bedrock_model(model: str) -> bool:
    """Check if model is an AWS Bedrock model."""
    return any(model.startswith(prefix) for prefix in BEDROCK_MODEL_PREFIXES)


# Model configurations
# supports_temp: whether the model supports temperature parameter
# supports_stop: whether the model supports stop sequences parameter
AZURE_MODELS = {
    # Sweden Central
    "gpt-4o": {"region": "sc", "supports_temp": True, "supports_stop": True},
    "o3": {"region": "sc", "supports_temp": False, "supports_stop": True},
    "o3-pro": {"region": "sc", "supports_temp": False, "supports_stop": True},
    "gpt-4.1": {"region": "sc", "supports_temp": True, "supports_stop": True},
    "gpt-5.1": {"region": "sc", "supports_temp": True, "supports_stop": False},
    "gpt-5.2": {"region": "sc", "supports_temp": True, "supports_stop": False},
    # East US 2 - GPT-5 models don't support stop sequences
    "gpt-5": {"region": "eus2", "supports_temp": False, "supports_stop": False},
    "gpt-5-codex": {"region": "eus2", "supports_temp": False, "supports_stop": False},
    "gpt-5-pro": {"region": "eus2", "supports_temp": False, "supports_stop": False},
}

AZURE_ENDPOINTS = {
    "sc": {
        "url": "https://regevlab-swedencentral-test.openai.azure.com/",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY_SC", ""),
    },
    "eus2": {
        "url": "https://regevlab-eastus2-test.openai.azure.com/",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY_EUS2", ""),
    },
}

AZURE_API_VERSION = "2025-04-01-preview"

# Cost rates (dollars per 1M tokens)
# NOTE: We ensure good support for the models listed here. Other models (e.g., o-mini, gemini-flash)
# are not guaranteed to be fully supported by the current implementation.
COST_RATES = {
    # OpenAI/Azure (2025+) - same models available via both endpoints
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-codex": {"input": 1.25, "output": 10.00},
    "gpt-5-pro": {"input": 2.50, "output": 20.00},
    "gpt-5.1": {"input": 2.00, "output": 8.00},
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "o3": {"input": 1.25, "output": 10.00},
    "o3-pro": {"input": 2.50, "output": 20.00},
    "gpt-4.1": {"input": 5.00, "output": 20.00},
    # Claude (latest)
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet-3-7-20250219": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
    # Gemini (available models)
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    # AWS Bedrock Claude models (same pricing as direct Anthropic API)
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.00, "output": 15.00},
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.00},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.80, "output": 4.00},
    "us.anthropic.claude-opus-4-5-20251101-v1:0": {"input": 15.00, "output": 75.00},
}


class CostCallback(BaseCallbackHandler):
    """Silent cost tracker - prints summary at the end."""

    def __init__(self, model: str):
        self.model = model
        self.rates = COST_RATES.get(model, {"input": 2.0, "output": 2.0})
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
        self.num_calls = 0

    def on_llm_end(self, response, **kwargs) -> None:
        """Track cost silently after each LLM call."""
        # Extract token usage from response (try multiple locations)
        # Note: getattr returns None if attribute exists but is None, so we use `or {}`
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {})

        # Also try generations[0].message.usage_metadata (for newer LangChain versions)
        if not usage and hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0]
            if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                metadata = gen.message.usage_metadata
                usage = {
                    "input_tokens": metadata.get("input_tokens", 0),
                    "output_tokens": metadata.get("output_tokens", 0),
                }

        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        # Update totals
        self.total_tokens["input"] += input_tokens
        self.total_tokens["output"] += output_tokens
        self.num_calls += 1

        # Calculate cost (rates are per 1M tokens)
        cost = (
            input_tokens * self.rates["input"] / 1_000_000 +
            output_tokens * self.rates["output"] / 1_000_000
        )
        self.total_cost += cost

    def print_summary(self) -> None:
        """Print cost summary at the end of conversation."""
        if self.num_calls > 0:
            print(f"\nCost Summary ({self.model})")
            print(f"Total calls:    {self.num_calls}")
            print(f"Input tokens:   {self.total_tokens['input']:,}")
            print(f"Output tokens:  {self.total_tokens['output']:,}")
            print(f"Total tokens:   {sum(self.total_tokens.values()):,}")
            print(f"Total cost:     ${self.total_cost:.4f}\n")


def make_llm(
    model: str,
    temperature: float = 0.5,
    streaming: bool = False,
    track_cost: bool = True,
    use_azure: bool = None,
    **kwargs
):
    """
    Create LLM instance. Supports OpenAI, Azure OpenAI, Anthropic, Bedrock, and Google Gemini.

    Configuration priority:
        1. AZURE_API_KEY + AZURE_API_ENDPOINT set → Azure OpenAI
        2. Model name detection (gemini-*, claude-*, gpt-*, etc.)

    Environment Variables:
        AZURE_API_KEY: Azure OpenAI API key
        AZURE_API_ENDPOINT: Azure OpenAI endpoint URL
        AZURE_DEPLOYMENT_NAME: Azure deployment name (defaults to model name)

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.5-pro")
        temperature: Sampling temperature (0-1). Ignored for reasoning models.
        streaming: Enable streaming responses
        track_cost: Enable cost tracking (default: True)
        **kwargs: Additional provider-specific parameters

    Returns:
        LangChain chat model instance
    """
    # Setup callbacks
    callbacks = []
    if streaming:
        callbacks.append(StreamingStdOutCallbackHandler())
    if track_cost:
        callbacks.append(CostCallback(model))

    # Check for custom OpenAI-compatible endpoint via environment variables
    custom_base_url = os.environ.get("CUSTOM_MODEL_BASE_URL", "")
    custom_api_key = os.environ.get("CUSTOM_MODEL_API_KEY", "EMPTY")

    # Custom OpenAI-compatible endpoint (LiteLLM, vLLM, Ollama, etc.)
    # Note: For Mistral models, the LiteLLM callback (local_llm/vllm/custom_callbacks.py)
    # handles continue_final_message automatically at the proxy layer.
    if custom_base_url:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=custom_base_url,
            api_key=custom_api_key if custom_api_key else "EMPTY",
            callbacks=callbacks,
            temperature=temperature,
            streaming=streaming,
            **kwargs
        )

    # Google Gemini (using OpenAI-compatible endpoint for consistent response format)
    if "gemini" in model:
        from langchain_openai import ChatOpenAI

        # Stop sequences for Gemini
        stop_sequences = kwargs.pop("stop_sequences", DEFAULT_STOP_SEQUENCES)

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            callbacks=callbacks,
            stop=stop_sequences,
            **kwargs
        )

    # AWS Bedrock models
    if _is_bedrock_model(model):
        try:
            from langchain_aws import ChatBedrockConverse
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "langchain_aws and boto3 packages required. "
                "Install with: pip install langchain-aws boto3"
            )

        # Get region from kwargs or use defaults
        kwargs.pop("profile_name", None)  # consume but ignore profile_name
        region_name = kwargs.pop("region_name", BedrockConfig.REGION)

        # Create boto3 client directly (uses default credential chain: env vars,
        # ~/.aws/credentials, instance profile — avoids SSO token expiration)
        bedrock_config = Config(
            read_timeout=300,  # 5 minutes for long responses
            connect_timeout=60,
            retries={"max_attempts": 3}
        )
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            config=bedrock_config,
        )

        # Stop sequences for Bedrock
        stop_sequences = kwargs.pop("stop_sequences", DEFAULT_STOP_SEQUENCES)

        # Use ChatBedrockConverse (newer API with proper stop_sequences support)
        return ChatBedrockConverse(
            model=model,
            client=bedrock_client,
            temperature=temperature,
            max_tokens=kwargs.pop("max_tokens", 8192),
            stop_sequences=stop_sequences,
            callbacks=callbacks,
            **kwargs
        )

    # Anthropic Claude (direct API)
    if "claude" in model:
        from langchain_anthropic import ChatAnthropic

        # Handle 1M context for Claude Sonnet 4.5 (beta)
        if "claude-sonnet-4-5" in model and kwargs.get("use_1m_context"):
            if "default_headers" not in kwargs:
                kwargs["default_headers"] = {}
            kwargs["default_headers"]["anthropic-beta"] = "context-1m-2025-08-07"
            kwargs.pop("use_1m_context")

        stop_sequences = kwargs.pop("stop_sequences", DEFAULT_STOP_SEQUENCES)

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=kwargs.pop("max_tokens", 8192),
            streaming=streaming,
            callbacks=callbacks,
            stop_sequences=stop_sequences,
            **kwargs
        )

    # Azure OpenAI (configured via environment variables)
    azure_api_key = os.environ.get("AZURE_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_API_ENDPOINT", "")

    if azure_api_key and azure_endpoint:
        from langchain_openai import AzureChatOpenAI

        azure_deployment = os.environ.get("AZURE_DEPLOYMENT_NAME", model)

        # Stop sequences prevent multiple action blocks and observation hallucination
        stop_sequences = kwargs.pop("stop_sequences", DEFAULT_STOP_SEQUENCES)

        # GPT-5 models only support temperature=1 (default) and don't support max_tokens or stop
        if any(x in model.lower() for x in NO_STOP_MODELS) or any(x in azure_deployment.lower() for x in NO_STOP_MODELS):
            return AzureChatOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                deployment_name=azure_deployment,
                api_version="2024-12-01-preview",
                streaming=streaming,
                callbacks=callbacks,
                temperature=1,  # GPT-5 only supports default value
                max_tokens=None,  # GPT-5 doesn't support this parameter
                # Note: GPT-5 doesn't support 'stop' parameter
                **kwargs
            )
        else:
            return AzureChatOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                deployment_name=azure_deployment,
                api_version="2024-10-21",
                streaming=streaming,
                callbacks=callbacks,
                temperature=temperature,
                stop=stop_sequences,
                **kwargs
            )

    # OpenAI vs Azure routing for GPT/O-series models
    if model.startswith(("gpt-", "o3", "o4")):
        # Default to Azure for models in AZURE_MODELS, otherwise direct OpenAI
        if use_azure is None:
            use_azure = model in AZURE_MODELS

        # Stop sequences prevent multiple action blocks and observation hallucination
        stop_sequences = kwargs.pop("stop_sequences", DEFAULT_STOP_SEQUENCES)

        if use_azure and model in AZURE_MODELS:
            # Azure OpenAI
            from langchain_openai import AzureChatOpenAI

            config = AZURE_MODELS[model]
            region = config["region"]
            endpoint = AZURE_ENDPOINTS[region]

            model_kwargs = {
                "azure_deployment": model,
                "azure_endpoint": endpoint["url"],
                "openai_api_key": endpoint["api_key"],
                "openai_api_version": AZURE_API_VERSION,
                "streaming": streaming,
                "callbacks": callbacks,
                **kwargs
            }

            if config.get("supports_stop", True):
                model_kwargs["stop"] = stop_sequences

            if config["supports_temp"]:
                model_kwargs["temperature"] = temperature

            return AzureChatOpenAI(**model_kwargs)
        else:
            # Direct OpenAI
            from langchain_openai import ChatOpenAI

            model_kwargs = {
                "model": model,
                "streaming": streaming,
                "callbacks": callbacks,
                **kwargs
            }

            # Some models don't support 'stop' parameter
            if not any(x in model.lower() for x in NO_STOP_MODELS):
                model_kwargs["stop"] = stop_sequences

            # O-series reasoning models don't support temperature
            if not model.startswith(("o3", "o4")):
                model_kwargs["temperature"] = temperature

            return ChatOpenAI(**model_kwargs)

    # Unknown model
    openai_azure_models = list(AZURE_MODELS.keys())
    claude_models = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"]
    gemini_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3-flash-preview"]
    bedrock_models = [
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
    ]

    supported_models = openai_azure_models + claude_models + gemini_models + bedrock_models

    raise ValueError(
        f"Model '{model}' not supported. "
        f"Supported models: {', '.join(sorted(supported_models))}"
    )


# Local embedding models (sentence-transformers)
LOCAL_EMBEDDING_MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",  # Best local model, on par with text-embedding-3-small
    "pubmedbert": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",  # Biomedical, 768 dim
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",  # Biomedical, 768 dim
}

# Default local model (qwen3-0.6b)
DEFAULT_LOCAL_EMBEDDING_MODEL = "qwen3-0.6b"


class LocalEmbeddings:
    """
    Local embedding model using sentence-transformers.

    Compatible with LangChain's Embeddings interface.
    No API rate limits, runs locally on CPU/GPU.
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL):
        """
        Initialize local embedding model.

        Args:
            model_name: Short name (e.g., "bge-large") or full HuggingFace model path
        """
        # Resolve short name to full path
        if model_name in LOCAL_EMBEDDING_MODELS:
            model_path = LOCAL_EMBEDDING_MODELS[model_name]
        else:
            model_path = model_name

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_path)
            self.model_name = model_path
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()


# Global cache for local embedding models (avoids reloading on every call)
_local_embedding_cache: dict[str, LocalEmbeddings] = {}


def make_llm_emb_local(model: str = DEFAULT_LOCAL_EMBEDDING_MODEL):
    """
    Create local embedding model using sentence-transformers.

    No API rate limits, runs locally. Good for high-volume embedding tasks.
    Models are cached globally to avoid reloading on every call.

    Args:
        model: Model name - either short name or full HuggingFace path
               Short names: "bge-large", "bge-base", "bge-small", "e5-large", "minilm", "mpnet"
               Default: "bge-large" (BAAI/bge-large-en-v1.5)

    Returns:
        LocalEmbeddings instance (LangChain compatible, cached globally)

    Example:
        emb = make_llm_emb_local("bge-large")
        vectors = emb.embed_documents(["text1", "text2"])
    """
    # Return cached model if already loaded
    if model in _local_embedding_cache:
        return _local_embedding_cache[model]

    # Load model and cache for reuse
    embeddings = LocalEmbeddings(model)
    _local_embedding_cache[model] = embeddings
    return embeddings


def make_llm_emb(
    model: str = "text-embedding-3-small",
    region: str = "eus2",
    use_local: bool = None,
    local_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
    input_type: str = None,
):
    """
    Create embedding model - supports Azure OpenAI, custom endpoints, or local models.

    Configuration priority (checked in order):
        1. use_local=True or USE_LOCAL_EMBEDDINGS env var → Local sentence-transformers
        2. CUSTOM_EMBED_BASE_URL → Custom OpenAI-compatible endpoint
        3. AZURE_API_KEY + AZURE_API_ENDPOINT → Azure OpenAI
        4. Hardcoded Azure endpoints (legacy fallback)

    Args:
        model: Azure/OpenAI embedding model name (default: text-embedding-3-small)
        region: Azure region - "eus2" (East US 2) or "sc" (Sweden Central)
        use_local: Force local embeddings (default: None = check env var)
        local_model: Local model to use if use_local=True (default: bge-large)
        input_type: Embedding input type - controls how queries vs documents are embedded.
                    For Cohere: "search_document" or "search_query"
                    For Qwen/vLLM: "query" or "document" (maps to prompt_name)
                    If None, not passed (for models that don't need it).

    Environment Variables:
        USE_LOCAL_EMBEDDINGS: Set to "true" to use local embeddings by default
        LOCAL_EMBEDDING_MODEL: Override local model name
        CUSTOM_EMBED_BASE_URL: Custom embedding endpoint URL
        CUSTOM_EMBED_API_KEY: API key for custom endpoint
        CUSTOM_EMBED_MODEL: Override model name for custom endpoint
        CUSTOM_EMBED_CHUNK_SIZE: Max texts per request for custom endpoint (default: 96)
        AZURE_API_KEY: Azure OpenAI API key
        AZURE_API_ENDPOINT: Azure OpenAI endpoint URL

    Returns:
        Embeddings instance (Azure, OpenAI, or Local)
    """
    # Check if local embeddings should be used
    if use_local is None:
        use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() != "false"

    if use_local:
        # Use local sentence-transformers model
        local_model_override = os.environ.get("LOCAL_EMBEDDING_MODEL", local_model)
        return make_llm_emb_local(local_model_override)

    from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

    # Check for custom embedding endpoint (e.g., LiteLLM proxy to Bedrock)
    custom_embed_url = os.environ.get("CUSTOM_EMBED_BASE_URL", "")
    custom_embed_key = os.environ.get("CUSTOM_EMBED_API_KEY", "")
    custom_embed_model = os.environ.get("CUSTOM_EMBED_MODEL", model)

    if custom_embed_url:
        # Build extra kwargs based on embedding model provider
        extra_body = {}
        model_lower = custom_embed_model.lower()

        # Default chunk size, can be overridden by env var
        env_chunk_size = os.environ.get("CUSTOM_EMBED_CHUNK_SIZE")

        match model_lower:
            # Qwen models: use prompt_name="query" for queries, nothing for documents
            # Max batch size depends on vLLM server config, default 256
            case m if "qwen" in m:
                if input_type in ("query", "search_query"):
                    extra_body = {"prompt_name": "query"}
                else:
                    extra_body = {}
                chunk_size = int(env_chunk_size) if env_chunk_size else 256

            # GTE models: use prompt_name="query" for queries, nothing for documents
            # Max batch size depends on vLLM server config, default 256
            case m if "gte" in m:
                if input_type in ("query", "search_query"):
                    extra_body = {"prompt_name": "query"}
                else:
                    extra_body = {}
                chunk_size = int(env_chunk_size) if env_chunk_size else 256

            # Cohere models: use input_type="search_query" or "search_document"
            # Max 96 texts per request for v3/v4
            case m if "cohere" in m:
                if input_type:
                    extra_body = {"input_type": input_type}
                else:
                    extra_body = {}
                chunk_size = int(env_chunk_size) if env_chunk_size else 96

            # OpenAI models: no extra params needed
            # Max ~2048 texts per request
            case m if "text-embedding" in m or "openai" in m:
                extra_body = {}
                chunk_size = int(env_chunk_size) if env_chunk_size else 2048

            # Default fallback: pass input_type if provided
            # Conservative default of 96
            case _:
                if input_type:
                    extra_body = {"input_type": input_type}
                else:
                    extra_body = {}
                chunk_size = int(env_chunk_size) if env_chunk_size else 96

        return OpenAIEmbeddings(
            model=custom_embed_model,
            openai_api_base=custom_embed_url,
            openai_api_key=custom_embed_key if custom_embed_key else "EMPTY",
            check_embedding_ctx_length=False,  # Required for Ollama compatibility
            chunk_size=chunk_size,
            model_kwargs={"extra_body": extra_body} if extra_body else {},
        )

    # Check for Azure configuration via environment variables
    azure_api_key = os.environ.get("AZURE_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_API_ENDPOINT", "")

    if azure_api_key and azure_endpoint:
        # Use Azure OpenAI embeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=model,
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2024-10-21",
        )

    # Fallback to hardcoded endpoints (legacy)
    endpoint = AZURE_ENDPOINTS[region]

    return AzureOpenAIEmbeddings(
        azure_deployment=model,
        azure_endpoint=endpoint["url"],
        openai_api_key=endpoint["api_key"],
        openai_api_version=AZURE_API_VERSION,
    )


def get_effective_embedding_model(model: str = "text-embedding-3-small") -> str:
    """
    Get the actual embedding model name that will be used by make_llm_emb().

    This is useful for caching - we need to know the actual model name to generate
    the correct cache key, since environment variables may override the default.

    Args:
        model: The requested model name (default: text-embedding-3-small)

    Returns:
        The actual model name that will be used:
        - If USE_LOCAL_EMBEDDINGS=true: returns LOCAL_EMBEDDING_MODEL or default local model
        - If CUSTOM_EMBED_MODEL set: returns that model name
        - Otherwise: returns the input model name

    Example:
        # Without env vars set
        get_effective_embedding_model("text-embedding-3-small")  # Returns "text-embedding-3-small"

        # With USE_LOCAL_EMBEDDINGS=true
        get_effective_embedding_model("text-embedding-3-small")  # Returns "qwen3-0.6b"
    """
    # Check if local embeddings are enabled
    use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "").lower() == "true"

    if use_local:
        # Return the local model name that will be used
        return os.environ.get("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)

    # Check for custom embedding model override
    custom_embed_model = os.environ.get("CUSTOM_EMBED_MODEL", "")
    if custom_embed_model:
        return custom_embed_model

    # Return the requested model (API model)
    return model
