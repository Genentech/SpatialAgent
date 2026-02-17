#!/usr/bin/env python3
"""Simple embeddings server using sentence-transformers."""

import argparse
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODELS = {
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-v2": "nomic-ai/nomic-embed-text-v2-moe",
    "qwen": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "qwen-small": "Alibaba-NLP/gte-Qwen2-0.5B-instruct",
}

app = FastAPI()
embed_model = None
model_name = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    data = [
        EmbeddingData(index=i, embedding=emb.tolist())
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=model_name,
        usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name}


def main():
    global embed_model, model_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", help="Model alias: qwen (recommended), nomic, nomic-v2, qwen-small")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    model_path = MODELS.get(args.model, args.model)
    model_name = args.model

    print(f"Loading embedding model: {model_path}")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(model_path, trust_remote_code=True)
    print(f"Ready on port {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
