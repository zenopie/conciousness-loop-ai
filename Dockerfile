FROM ubuntu:22.04

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    nodejs \
    npm \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

# Add claude to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install claude-max
RUN pip3 install claude-max

# Copy source code to workspace - Claude can read AND modify this
COPY src/claude_loop.py ./claude_loop.py

EXPOSE 8080

# Run from workspace where Claude has full access to its own source
CMD ["python3", "-u", "claude_loop.py"]
