ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    gcc cmake build-essential \
    g++ libffi-dev \
    git \
    curl \
    wget \
    patch \
    bash \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --recurse-submodules https://github.com/angosr/AgentGym.git AgentGym \
    && cd AgentGym \
    && git checkout b0afbc19504ead9bbdffa0e070a061a3b89879f2 \
    && git submodule update --init --recursive

ENV PATH=/opt/miniconda/envs/agentenv/bin:/opt/miniconda/bin:$PATH

ARG ENV_NAME
ENV ENV_NAME=${ENV_NAME}

COPY preprocess_env.sh /app/
RUN chmod +x /app/preprocess_env.sh && /app/preprocess_env.sh

RUN pip install /app/AgentGym/agentenv

COPY postprocess_env.sh /app/
RUN chmod +x /app/postprocess_env.sh && /app/postprocess_env.sh

ENV ALFWORLD_DATA=/root/.cache/alfworld

ARG TOOL_NAME
ENV TOOL_NAME=${TOOL_NAME}
COPY env.py /app/

HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=2 \
  CMD sh -c "curl -f http://localhost:${AFFINETES_PORT:-8000}/health || exit 1"

WORKDIR /sandbox
# IMPORTANT: Must use single worker mode (no --workers or --workers 1)
# Reason: AgentGym evaluator creates stateful agents that callback to the same server
# Multiple workers would cause requests to be routed to different processes, losing state
# For better performance within single process, use asyncio event loop
CMD uvicorn env:app \
  --host 0.0.0.0 \
  --port ${AFFINETES_PORT:-8000} \
  --app-dir /app \
  --loop asyncio \
  --workers 1