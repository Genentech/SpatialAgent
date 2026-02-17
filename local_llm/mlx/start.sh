#!/bin/bash
# Start MLX servers for SpatialAgent (Apple Silicon)
#
# Usage:
#   ./local_llm/mlx/start.sh        # Start all servers
#   ./local_llm/mlx/start.sh stop   # Stop all servers
#   ./local_llm/mlx/start.sh status # Check server status

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_LLM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$LOCAL_LLM_DIR")"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv-mlx"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"

# Server ports
MLX_VLM_PORT=8081
MLX_LM_PORT=8083
EMBED_PORT=8082
LITELLM_PORT=8080

mkdir -p "$LOG_DIR" "$PID_DIR"

start_servers() {
    echo "=============================================="
    echo "  Starting MLX Servers (Apple Silicon)"
    echo "=============================================="
    echo ""

    # Check venv exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: Virtual environment not found at $VENV_DIR"
        echo "Run ./local_llm/mlx/setup.sh first."
        exit 1
    fi

    source "$VENV_DIR/bin/activate"

    # 1. MLX VLM Server (Vision models)
    echo "Starting mlx_vlm server on port $MLX_VLM_PORT..."
    python -m mlx_vlm.server --port $MLX_VLM_PORT > "$LOG_DIR/mlx_vlm.log" 2>&1 &
    echo $! > "$PID_DIR/mlx_vlm.pid"

    # 2. MLX LM Server (Text-only models)
    echo "Starting MLX LM server on port $MLX_LM_PORT..."
    python "$LOCAL_LLM_DIR/mlx/lm_server.py" --port $MLX_LM_PORT > "$LOG_DIR/mlx_lm.log" 2>&1 &
    echo $! > "$PID_DIR/mlx_lm.pid"

    # 3. Embeddings Server
    echo "Starting embedding server on port $EMBED_PORT..."
    python "$LOCAL_LLM_DIR/mlx/embed_server.py" --model qwen --port $EMBED_PORT > "$LOG_DIR/embed.log" 2>&1 &
    echo $! > "$PID_DIR/embed.pid"

    # Wait for backend servers to initialize
    echo "Waiting for backend servers..."
    sleep 5

    # 4. LiteLLM Proxy
    echo "Starting LiteLLM proxy on port $LITELLM_PORT..."
    litellm --config "$LOCAL_LLM_DIR/mlx/litellm_config.yaml" --port $LITELLM_PORT > "$LOG_DIR/litellm.log" 2>&1 &
    echo $! > "$PID_DIR/litellm.pid"

    sleep 3

    echo ""
    echo "=============================================="
    echo "  All servers started!"
    echo "=============================================="
    echo ""
    echo "Endpoints:"
    echo "  LiteLLM Proxy:  http://localhost:$LITELLM_PORT"
    echo "  MLX VLM:        http://localhost:$MLX_VLM_PORT"
    echo "  MLX LM:         http://localhost:$MLX_LM_PORT"
    echo "  Embeddings:     http://localhost:$EMBED_PORT"
    echo ""
    echo "To use with SpatialAgent:"
    echo ""
    echo "  export CUSTOM_MODEL_BASE_URL=http://localhost:$LITELLM_PORT/v1"
    echo "  export CUSTOM_EMBED_BASE_URL=http://localhost:$LITELLM_PORT/v1"
    echo "  export CUSTOM_EMBED_MODEL=qwen"
    echo ""
    echo "Logs: $LOG_DIR/"
    echo "Stop: ./local_llm/mlx/start.sh stop"
}

stop_servers() {
    echo "Stopping MLX servers..."

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

    echo "All servers stopped."
}

status_servers() {
    echo "=============================================="
    echo "  MLX Server Status"
    echo "=============================================="
    echo ""

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
    echo "Port check:"
    for port in $LITELLM_PORT $MLX_VLM_PORT $MLX_LM_PORT $EMBED_PORT; do
        if lsof -i :$port >/dev/null 2>&1; then
            echo "  :$port - in use"
        else
            echo "  :$port - free"
        fi
    done
}

case "${1:-start}" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    status)
        status_servers
        ;;
    restart)
        stop_servers
        sleep 2
        start_servers
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
