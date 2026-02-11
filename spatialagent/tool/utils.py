"""Utility functions for tool module."""

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Cache directory for embeddings (under repo for reproducibility)
_EMBEDDING_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "embedding_cache"


def _get_cache_key(database: str, embedding_model: str, n_items: int) -> str:
    """Generate a cache key from database name, embedding model, and item count.

    Args:
        database: Database identifier (e.g., 'panglao_Hs', 'cellmarker2_Human')
        embedding_model: Embedding model name (e.g., 'text-embedding-3-small', 'Qwen/Qwen3-Embedding-0.6B')
        n_items: Number of items in the database

    Returns:
        Cache key string: {database}_{embedding_model}_{n_items}

    Examples:
        - OpenAI API: 'panglao_Hs_text-embedding-3-small_256'
        - Local Qwen: 'panglao_Hs_Qwen-Qwen3-Embedding-0.6B_256'
    """
    # Sanitize model name for filename (replace slashes, etc.)
    model_safe = embedding_model.replace("/", "-").replace(":", "-")
    return f"{database}_{model_safe}_{n_items}"


def _load_cached_embeddings(cache_key: str) -> np.ndarray | None:
    """Load cached embeddings if they exist."""
    cache_file = _EMBEDDING_CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cached embeddings: {e}")
    return None


def _save_cached_embeddings(cache_key: str, embeddings: np.ndarray) -> None:
    """Save embeddings to cache."""
    _EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _EMBEDDING_CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        logging.warning(f"Failed to save embeddings to cache: {e}")


def _embed_with_retry(embedder, texts: list, max_retries: int = 3, base_delay: float = 10.0) -> np.ndarray:
    """Embed texts with retry logic for rate limit errors.

    Args:
        embedder: LangChain embeddings object
        texts: List of texts to embed
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        numpy array of embeddings
    """
    for attempt in range(max_retries + 1):
        try:
            return np.array(embedder.embed_documents(texts))
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "429" in error_str or "RateLimit" in error_str or "rate" in error_str.lower():
                if attempt < max_retries:
                    # Extract wait time from error message if available
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    if "retry after" in error_str.lower():
                        try:
                            # Try to parse "retry after X seconds" from error
                            import re
                            match = re.search(r"retry after (\d+)", error_str.lower())
                            if match:
                                wait_time = max(int(match.group(1)), wait_time)
                        except:
                            pass
                    logging.warning(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
            # Not a rate limit error or max retries reached
            raise
    # Should not reach here, but just in case
    raise RuntimeError(f"Failed after {max_retries} retries")


def find_most_similar(llm_emb_query, queries, descriptions, batch_size=1000, llm_emb_doc=None,
                      database: str = None, embedding_model: str = None):
    """Process queries and descriptions in batches and return matches with similarities.

    Args:
        llm_emb_query: Embedder for queries (with input_type="search_query" for Cohere)
        queries: List of query strings
        descriptions: List of description strings to match against
        batch_size: Batch size for processing descriptions
        llm_emb_doc: Embedder for documents (with input_type="search_document" for Cohere).
                     If None, uses llm_emb_query for both (backwards compatible).
        database: Database identifier for caching (e.g., 'panglao_Hs', 'cellmarker2_Human')
        embedding_model: Embedding model name for caching (e.g., 'text-embedding-3-small')
    """
    # Use separate embedder for docs if provided, otherwise use same for both
    if llm_emb_doc is None:
        llm_emb_doc = llm_emb_query

    # Embed queries with retry (usually smaller, so process at once)
    query_embeddings = _embed_with_retry(llm_emb_query, queries)

    # Check for cached description embeddings
    description_embeddings = None
    cache_key = None
    if database and embedding_model:
        cache_key = _get_cache_key(database, embedding_model, len(descriptions))
        description_embeddings = _load_cached_embeddings(cache_key)
        if description_embeddings is not None:
            logging.info(f"Loaded cached embeddings for {database} ({len(descriptions)} items)")

    # If not cached, compute embeddings in batches
    if description_embeddings is None:
        description_embeddings_list = []

        # Calculate number of batches needed
        num_batches = int(np.ceil(len(descriptions) / batch_size))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(descriptions))

            # Get current batch of descriptions
            desc_batch = descriptions[start_idx:end_idx]

            # Embed current batch using document embedder with retry
            batch_embeddings = _embed_with_retry(llm_emb_doc, desc_batch)
            description_embeddings_list.extend(batch_embeddings)

        description_embeddings = np.array(description_embeddings_list)

        # Cache the embeddings if cache_key provided
        if cache_key:
            _save_cached_embeddings(cache_key, description_embeddings)
            logging.info(f"Cached embeddings for {database} ({len(descriptions)} items)")

    # Find matches for each query
    matched_descriptions = []
    logging.info("\n{:<40} | {:<40} | {:<10}".format("Query", "Best Match", "Similarity"))
    logging.info("-" * 97)

    for i, query in enumerate(queries):
        similarities = cosine_similarity([query_embeddings[i]], description_embeddings)[0]
        most_similar_idx = np.argmax(similarities)

        logging.info("{:<40} | {:<40} | {:.3f}".format(
            query[:40],
            descriptions[most_similar_idx][:40],
            similarities[most_similar_idx]
        ))
        matched_descriptions.append(descriptions[most_similar_idx])

    return matched_descriptions


def parse_list_string(input_str: str, uppercase: bool = False) -> list[str]:
    """
    Parse a comma-separated string that may be a stringified Python list.

    Handles various input formats from LLM tool calls:
    - "gene1, gene2, gene3" -> ["gene1", "gene2", "gene3"]
    - "['gene1', 'gene2']" -> ["gene1", "gene2"]
    - '["gene1", "gene2"]' -> ["gene1", "gene2"]
    - "gene1" -> ["gene1"]

    Args:
        input_str: Input string to parse
        uppercase: If True, convert all items to uppercase (useful for genes)

    Returns:
        List of cleaned strings
    """
    if not input_str or not input_str.strip():
        return []

    cleaned = input_str.strip()

    # Remove outer brackets if present (stringified list)
    if (cleaned.startswith('[') and cleaned.endswith(']')) or \
       (cleaned.startswith('(') and cleaned.endswith(')')):
        cleaned = cleaned[1:-1]

    # Split by comma and clean each item
    items = []
    for item in cleaned.split(","):
        # Strip whitespace, quotes, and any remaining brackets
        item_cleaned = item.strip().strip("'\"[]()").strip()
        if item_cleaned:
            if uppercase:
                item_cleaned = item_cleaned.upper()
            items.append(item_cleaned)

    return items


def clean_code(code):
    """Clean code by removing markdown and main() blocks."""
    # Remove markdown if present
    code = code.replace('```python\n', '').replace('```', '').strip()

    # Remove main() and if __name__ == "__main__" block
    lines = code.split('\n')
    cleaned_lines = []
    skip_block = False
    for line in lines:
        if 'def main()' in line or 'if __name__' in line:
            skip_block = True
            continue
        if skip_block and line.startswith((' ', '\t')):
            continue
        if not line.strip():
            skip_block = False
        if not skip_block:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
