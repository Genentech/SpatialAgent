#!/usr/bin/env python3
"""Dynamic mlx_lm server that loads models on-demand like mlx_vlm."""

import argparse
import asyncio
import gc
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlx_lm
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

app = FastAPI(title="MLX-LM Dynamic Server")

# Global state
current_model_name = None
model = None
tokenizer = None
last_request_time = 0
UNLOAD_TIMEOUT = 60  # seconds


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[Choice]
    usage: Usage


def collate_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role (like Ollama does).

    This fixes compatibility with models like Mistral/Ministral whose chat
    templates don't allow multiple consecutive messages with the same role.
    """
    if not messages:
        return messages

    collated = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["role"] == collated[-1]["role"]:
            collated[-1]["content"] += "\n\n" + msg["content"]
        else:
            collated.append(msg.copy())
    return collated


def unload_model():
    """Unload current model and free memory."""
    global current_model_name, model, tokenizer
    if current_model_name:
        print(f"Unloading model: {current_model_name}")
        model = None
        tokenizer = None
        current_model_name = None
        gc.collect()
        mx.metal.clear_cache()
        print("Model unloaded and cache cleared.")


def load_model(model_name: str):
    """Load a model, unloading any existing one first."""
    global current_model_name, model, tokenizer, last_request_time

    last_request_time = time.time()

    if current_model_name == model_name:
        print(f"Using cached model: {model_name}")
        return

    if current_model_name:
        unload_model()

    print(f"Loading model: {model_name}")
    model, tokenizer = mlx_lm.load(model_name, tokenizer_config={"trust_remote_code": True})
    current_model_name = model_name
    print("Model loaded successfully.")


async def auto_unload_task():
    """Background task to unload model after timeout."""
    global last_request_time
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        if current_model_name and (time.time() - last_request_time) > UNLOAD_TIMEOUT:
            print(f"Auto-unloading model after {UNLOAD_TIMEOUT}s of inactivity")
            unload_model()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_unload_task())


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        load_model(request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Build conversation and collate consecutive same-role messages (Ollama-style)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    messages = collate_messages(messages)

    try:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate with sampler
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Count tokens (approximate)
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(response))

        return ChatResponse(
            id="chatcmpl-mlx",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": current_model_name}


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": current_model_name or "none", "object": "model"}]}


@app.post("/unload")
async def unload():
    """Manually unload the current model."""
    if current_model_name:
        unload_model()
        return {"status": "unloaded"}
    return {"status": "no model loaded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--timeout", type=int, default=60, help="Auto-unload timeout in seconds")
    args = parser.parse_args()

    UNLOAD_TIMEOUT = args.timeout
    print(f"Starting MLX-LM dynamic server on {args.host}:{args.port}")
    print(f"Auto-unload timeout: {UNLOAD_TIMEOUT}s")
    uvicorn.run(app, host=args.host, port=args.port)
