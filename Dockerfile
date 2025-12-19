FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install unsloth without deps (uses base image's torch + above packages)
RUN pip install --no-cache-dir --no-deps unsloth unsloth_zoo

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
