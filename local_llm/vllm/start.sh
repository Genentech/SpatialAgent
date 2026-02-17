#!/bin/bash
# Start vLLM servers for SpatialAgent (NVIDIA GPUs)
#
# Usage:
#   ./local_llm/vllm/start.sh              # Start all servers (default: qwen3-vl-32b)
#   ./local_llm/vllm/start.sh ministral    # Start with Ministral-3-14B
#   ./local_llm/vllm/start.sh stop         # Stop all servers
#   ./local_llm/vllm/start.sh status       # Check server status
#
# Environment variables:
#   CHAT_MODEL    - Override chat model (default: Qwen/Qwen3-VL-32B-Instruct)
#   EMBED_MODEL   - Override embedding model (default: Qwen/Qwen3-Embedding-0.6B)
#   TP_SIZE       - Tensor parallel size (default: auto-detect)
#   MAX_MODEL_LEN - Max context length (default: 131072)
#   LITELLM_PORT  - LiteLLM proxy port (default: 8088)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_LLM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$LOCAL_LLM_DIR")"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"

# Default configuration
CHAT_PORT=8000
EMBED_PORT=8001
LITELLM_PORT=${LITELLM_PORT:-8088}

# Model defaults
EMBED_MODEL=${EMBED_MODEL:-"Qwen/Qwen3-Embedding-0.6B"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}

# Auto-detect GPU count for tensor parallelism
detect_tp_size() {
    local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    # Use all available GPUs for maximum speed (up to 4)
    if [ "$gpu_count" -ge 4 ]; then
        echo 4
    elif [ "$gpu_count" -ge 2 ]; then
        echo 2
    else
        echo 1
    fi
}

TP_SIZE=${TP_SIZE:-$(detect_tp_size)}

mkdir -p "$LOG_DIR" "$PID_DIR"

start_servers() {
    local model_choice="${1:-qwen}"

    # Set chat model based on choice
    case "$model_choice" in
        ministral|ministral-3)
            CHAT_MODEL=${CHAT_MODEL:-"mistralai/Ministral-3-14B-Instruct-2512"}
            EXTRA_ARGS="--tokenizer_mode mistral --config_format mistral --load_format mistral"
            ;;
        ministral-reasoning|ministral-3-reasoning)
            CHAT_MODEL=${CHAT_MODEL:-"mistralai/Ministral-3-14B-Reasoning-2512"}
            EXTRA_ARGS="--tokenizer_mode mistral --config_format mistral --load_format mistral"
            ;;
        qwen|qwen3-vl|qwen3-vl-32b|*)
            CHAT_MODEL=${CHAT_MODEL:-"Qwen/Qwen3-VL-32B-Instruct"}
            EXTRA_ARGS=""
            ;;
    esac

    echo "=============================================="
    echo "  Starting vLLM Servers"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  Chat model:     $CHAT_MODEL"
    echo "  Embed model:    $EMBED_MODEL"
    echo "  Tensor parallel: $TP_SIZE GPUs"
    echo "  Max context:    $MAX_MODEL_LEN tokens"
    echo ""

    # Check environments exist
    if [ ! -d ".venv-vllm" ] || [ ! -d ".venv-litellm" ]; then
        echo "Error: Environments not found. Run ./local_llm/vllm/setup.sh first."
        exit 1
    fi

    # 1. Start embedding server
    echo "Starting embedding server on port $EMBED_PORT..."
    .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$EMBED_MODEL" \
        --port $EMBED_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.02 \
        --max-model-len 512 \
        --enforce-eager > "$LOG_DIR/vllm_embed.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_embed.pid"

    # Wait for embedding server
    echo "  Waiting for embedding server to initialize..."
    for i in {1..60}; do
        if curl -s http://localhost:$EMBED_PORT/v1/models > /dev/null 2>&1; then
            echo "  Embedding server ready!"
            break
        fi
        sleep 2
    done

    # 2. Start chat server
    echo ""
    echo "Starting chat server on port $CHAT_PORT..."
    echo "  (This may take 2-5 minutes for model loading and CUDA graph compilation)"
    .venv-vllm/bin/vllm serve "$CHAT_MODEL" \
        --port $CHAT_PORT \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.90 \
        $EXTRA_ARGS > "$LOG_DIR/vllm_chat.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_chat.pid"

    # Wait for chat server
    echo "  Waiting for chat server to initialize..."
    for i in {1..180}; do
        if curl -s http://localhost:$CHAT_PORT/v1/models > /dev/null 2>&1; then
            echo "  Chat server ready!"
            break
        fi
        # Show progress
        if [ $((i % 30)) -eq 0 ]; then
            echo "  Still loading... (${i}s)"
        fi
        sleep 2
    done

    # 3. Start LiteLLM proxy
    echo ""
    echo "Starting LiteLLM proxy on port $LITELLM_PORT..."
    .venv-litellm/bin/litellm \
        --config "$LOCAL_LLM_DIR/vllm/litellm_config.yaml" \
        --port $LITELLM_PORT > "$LOG_DIR/litellm.log" 2>&1 &
    echo $! > "$PID_DIR/litellm.pid"

    sleep 5

    # Verify all servers
    echo ""
    echo "=============================================="
    echo "  Server Status"
    echo "=============================================="

    local all_ok=true

    if curl -s http://localhost:$EMBED_PORT/v1/models > /dev/null 2>&1; then
        echo "  ✓ Embedding server (port $EMBED_PORT)"
    else
        echo "  ✗ Embedding server (port $EMBED_PORT) - FAILED"
        all_ok=false
    fi

    if curl -s http://localhost:$CHAT_PORT/v1/models > /dev/null 2>&1; then
        echo "  ✓ Chat server (port $CHAT_PORT)"
    else
        echo "  ✗ Chat server (port $CHAT_PORT) - FAILED"
        all_ok=false
    fi

    if curl -s http://localhost:$LITELLM_PORT/v1/models > /dev/null 2>&1; then
        echo "  ✓ LiteLLM proxy (port $LITELLM_PORT)"
    else
        echo "  ✗ LiteLLM proxy (port $LITELLM_PORT) - FAILED"
        all_ok=false
    fi

    echo ""
    if [ "$all_ok" = true ]; then
        echo "=============================================="
        echo "  All servers running!"
        echo "=============================================="
        echo ""
        echo "To use with SpatialAgent:"
        echo ""
        echo "  export CUSTOM_MODEL_BASE_URL=http://localhost:$LITELLM_PORT/v1"
        echo "  export CUSTOM_EMBED_BASE_URL=http://localhost:$LITELLM_PORT/v1"
        echo "  export CUSTOM_EMBED_MODEL=qwen3-embedding"
        echo "  export TOKENIZERS_PARALLELISM=false"
        echo ""
        echo "Then in Python:"
        echo ""
        echo "  from spatialagent.agent import SpatialAgent, make_llm"
        echo "  llm = make_llm('qwen3-vl-32b')"
        echo "  agent = SpatialAgent(llm=llm, save_path='./experiments/local/')"
        echo ""
        echo "Logs: $LOG_DIR/"
        echo "Stop:  ./local_llm/vllm/start.sh stop"
    else
        echo "Some servers failed to start. Check logs in $LOG_DIR/"
    fi
}

stop_servers() {
    echo "Stopping vLLM servers..."

    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            name=$(basename "$pid_file" .pid)
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Stopping $name (PID $pid)..."
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done

    # Also kill any orphaned processes on the ports
    for port in $CHAT_PORT $EMBED_PORT $LITELLM_PORT; do
        lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done

    echo "All servers stopped."
}

status_servers() {
    echo "=============================================="
    echo "  vLLM Server Status"
    echo "=============================================="
    echo ""

    # Check PIDs
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            name=$(basename "$pid_file" .pid)
            if kill -0 "$pid" 2>/dev/null; then
                echo "  $name: running (PID $pid)"
            else
                echo "  $name: stopped (stale PID file)"
            fi
        fi
    done

    echo ""
    echo "Port status:"
    for port in $CHAT_PORT $EMBED_PORT $LITELLM_PORT; do
        if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
            echo "  :$port - responding"
        elif lsof -i :$port > /dev/null 2>&1; then
            echo "  :$port - in use (not responding to /v1/models)"
        else
            echo "  :$port - free"
        fi
    done

    echo ""
    echo "GPU memory:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
}

# Main
case "${1:-start}" in
    start|qwen|qwen3-vl|qwen3-vl-32b)
        start_servers "qwen"
        ;;
    ministral|ministral-3)
        start_servers "ministral"
        ;;
    ministral-reasoning|ministral-3-reasoning)
        start_servers "ministral-reasoning"
        ;;
    stop)
        stop_servers
        ;;
    status)
        status_servers
        ;;
    restart)
        stop_servers
        sleep 3
        start_servers "${2:-qwen}"
        ;;
    *)
        echo "Usage: $0 {start|ministral|ministral-reasoning|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start              Start with Qwen3-VL-32B (default)"
        echo "  ministral          Start with Ministral-3-14B-Instruct"
        echo "  ministral-reasoning Start with Ministral-3-14B-Reasoning"
        echo "  stop               Stop all servers"
        echo "  status             Check server status"
        echo "  restart            Restart all servers"
        exit 1
        ;;
esac
