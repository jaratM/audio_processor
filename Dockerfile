FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install psycopg2-binary

COPY . .

# Default config path can be overridden at runtime
ENV CONFIG_FILE=config.yaml

# Create mount points for data and logs
VOLUME ["/app/input", "/app/output", "/app/logs"]

# Entry point runs the orchestrator; pass args via CMD
ENTRYPOINT ["python3", "run.py", "--config", "${CONFIG_FILE}"]
CMD ["--save-mode", "database", "--performance-report"]


