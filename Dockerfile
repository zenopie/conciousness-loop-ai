FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install unsloth with dependencies
RUN pip install --no-cache-dir unsloth

# Copy application code
COPY src/ ./src/

# Create workspace directory for file executor
RUN mkdir -p /home/claude/loop/workspace

# Create input file
RUN touch input.txt

# Create empty context.txt (can be overwritten via mount or CUSTOM_CONTEXT env var)
RUN touch /app/context.txt

EXPOSE 8080

CMD ["python", "-u", "src/server.py"]
