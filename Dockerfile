FROM ubuntu:22.04

# Create non-root user (Claude Code blocks --dangerously-skip-permissions for root)
RUN useradd -m -s /bin/bash claude && \
    mkdir -p /workspace && \
    chown -R claude:claude /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    nodejs \
    npm \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Give claude sudo access
RUN echo "claude ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to claude user
USER claude
WORKDIR /workspace

# Install Claude Code as claude user
RUN curl -fsSL https://claude.ai/install.sh | bash

# Add claude to PATH
ENV PATH="/home/claude/.local/bin:$PATH"

# Install claude-max
RUN pip3 install --user claude-max

# Copy source code to workspace - Claude can read AND modify this
COPY --chown=claude:claude src/claude_loop.py ./claude_loop.py

EXPOSE 8080

# Run from workspace where Claude has full access to its own source
CMD ["python3", "-u", "claude_loop.py"]
