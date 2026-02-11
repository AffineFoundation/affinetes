# Codex Django Bug-Fix Environment
# 
# This is a FUNCTION-BASED environment. Affinetes will automatically
# inject the HTTP server during the two-stage build process.
#
# Contains:
# - Python 3.12 with required packages
# - OpenAI Codex CLI for bug injection (via LLM)
# - Django source code (stable/5.1.x) for testing

FROM python:3.12-slim

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    curl \
    gnupg \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (required for Codex CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install OpenAI Codex CLI globally
RUN npm install -g @openai/codex

# Verify Codex installation
RUN codex --version || echo "Codex installed"

# Clone Django repository (specific stable version)
RUN git clone --depth 1 --branch stable/5.1.x https://github.com/django/django.git /django

# Set up Django for running tests
WORKDIR /django

# Install Django dependencies for running tests
RUN pip install --no-cache-dir \
    asgiref \
    sqlparse \
    tblib \
    pytz \
    pytest \
    pytest-django \
    coverage \
    pillow \
    pyyaml \
    argon2-cffi \
    bcrypt \
    docutils \
    jinja2 \
    numpy \
    selenium \
    redis \
    pymemcache

# Install Django in development mode
RUN pip install -e .

# Set working directory for the environment
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Create Codex config directory
RUN mkdir -p /root/.codex

# Set environment variables
ENV CODEX_LOG_LEVEL=INFO
ENV DJANGO_SETTINGS_MODULE=django.conf.global_settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# NOTE: No CMD specified - affinetes will inject the HTTP server
# during the two-stage build process for function-based environments
