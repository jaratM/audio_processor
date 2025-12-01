FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS base

# Évite les prompts interactifs pendant l'installation

ENV DEBIAN_FRONTEND=noninteractive

# Met à jour le système et installe les dépendances système

 

# Edit Sources

RUN sed -i 's/http/https/g' /etc/apt/sources.list

 

RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Met à jour pip

RUN pip3 install --upgrade pip

# Définit le répertoire de travail

WORKDIR /app

# Copie le fichier requirements.txt en premier (pour le cache Docker)

# Create necessary directories

RUN mkdir -p models logs

COPY requirements.txt .

# Installe les dépendances Python

RUN pip3 install --no-cache-dir -r requirements.txt

# Copie le code source (en excluant models/ grâce au .dockerignore)

COPY . .

# Crée les répertoires pour les volumes si nécessaire

RUN mkdir -p /app/models /app/logs && chmod +x run_cron.sh

# Default config path can be overridden at runtime

ENV CONFIG_FILE=config.yaml

# Create mount points for data and logs

VOLUME ["/app/input", "/app/output", "/app/logs"]

# Entry point runs the orchestrator; pass args via CMD

ENTRYPOINT ["/bin/bash", "-c", "python3 run.py --config \"$CONFIG_FILE\" \"$@\"", "--"]

CMD ["--save-mode", "database", "--performance-report", "--load-metadata"]