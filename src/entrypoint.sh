#!/bin/bash
# Entrypoint that allows hot-reload of the Python script
# Touch /workspace/RESTART to trigger a restart

cd /workspace

while true; do
    echo "Starting claude_loop.py..."
    python3 -u claude_loop.py &
    PID=$!

    # Watch for restart signal
    while kill -0 $PID 2>/dev/null; do
        if [ -f /workspace/RESTART ]; then
            echo "Restart signal received, reloading..."
            rm -f /workspace/RESTART
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null
            break
        fi
        sleep 1
    done

    # If process died without restart signal, wait before restarting
    if ! [ -f /workspace/RESTART ]; then
        wait $PID
        EXIT_CODE=$?
        echo "Process exited with code $EXIT_CODE, restarting in 5s..."
        sleep 5
    fi
done
