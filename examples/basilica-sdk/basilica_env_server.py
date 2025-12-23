#!/usr/bin/env python3
"""
Basilica Environment Server - Deploy affinetes environment on Basilica

Deploys an affinetes evaluation environment on Basilica infrastructure.
The environment exposes HTTP endpoints for evaluation tasks.

Usage:
    export BASILICA_API_TOKEN="your-token"

    # Deploy with default image (affine-env:v4 - has HTTP server built-in)
    python basilica_env_server.py

    # Deploy with custom image
    ENV_IMAGE="docker.io/affinefoundation/lgc:pi" python basilica_env_server.py
"""

import os
import sys
import time

from basilica import BasilicaClient
from basilica.deployment import Deployment


# Configuration from environment
ENV_IMAGE = os.environ.get("ENV_IMAGE", "affinefoundation/affine-env:v4")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", "affine-env")
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))


def main():
    """Deploy environment server and print connection info."""
    api_token = os.environ.get("BASILICA_API_TOKEN")
    if not api_token:
        print("Error: BASILICA_API_TOKEN environment variable not set")
        print("Please set: export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)

    print("=" * 60)
    print("Basilica Environment Server Deployment")
    print("=" * 60)
    print(f"Image: {ENV_IMAGE}")
    print(f"Deployment Name: {DEPLOYMENT_NAME}")
    print()
    print("Deploying to Basilica...")

    client = BasilicaClient()

    # Use low-level API to specify command for starting the HTTP server
    # The affinetes images need uvicorn to be started explicitly
    response = client.create_deployment(
        instance_name=DEPLOYMENT_NAME,
        image=ENV_IMAGE,
        port=8000,
        cpu="2000m",
        memory="8Gi",
        ttl_seconds=3600,
        public=True,
        # Start uvicorn server - the _affinetes module is injected during affinetes build
        command=["python", "-m", "uvicorn"],
        args=[
            "_affinetes.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
    )

    print(f"Deployment created: {response.instance_name}")
    print("Waiting for deployment to be ready...")

    # Wait for deployment to be ready
    deployment = Deployment._from_response(client, response)
    deployment.wait_until_ready(timeout=TIMEOUT)
    deployment.refresh()

    print()
    print("=" * 60)
    print("Deployment Ready")
    print("=" * 60)
    print(f"Deployment Name: {deployment.name}")
    print(f"Base URL: {deployment.url}")
    print()
    print("Use this URL with basilica_evaluator.py:")
    print(f"  --env-url {deployment.url}")
    print()
    print("Test health:")
    print(f"  curl {deployment.url}/health")

    return deployment


if __name__ == "__main__":
    main()
