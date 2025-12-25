# /// script
# dependencies = [
#   "basilica-sdk>=0.10.0",
#   "affinetes",
# ]
# ///

"""
Basilica LLM Server - Deploy vLLM service on Basilica GPU

Deploys an OpenAI-compatible LLM endpoint on Basilica's GPU infrastructure.
The endpoint can be used with affinetes evaluation framework.

Usage:
    export BASILICA_API_TOKEN="your-token"

    # Deploy LLM server
    python basilica_llm_server.py

    # Deploy with custom model
    MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" python basilica_llm_server.py

    # Delete deployment when done
    python basilica_llm_server.py --delete

    # Delete specific deployment by name
    python basilica_llm_server.py --delete --name my-deployment-id
"""

import argparse
import os
import sys

from basilica import BasilicaClient
from basilica.decorators import deployment


# Configuration from environment
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", "affine-llm")


@deployment(
    name=DEPLOYMENT_NAME,
    image="vllm/vllm-openai:latest",
    gpu="A100",
    gpu_count=1,
    memory="40Gi",
    port=8000,
    ttl_seconds=3600,
    timeout=600,
)
def serve_llm():
    """
    Deploy vLLM service with OpenAI-compatible API.

    GPU detection is automatic. Model is configured via environment variable.
    """
    import os
    import subprocess

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    cmd = [
        "vllm", "serve", model_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", model_name,
    ]

    subprocess.Popen(cmd).wait()


def delete_deployment(name: str) -> None:
    """Delete a deployment by name."""
    client = BasilicaClient()
    print(f"Deleting deployment: {name}")
    client.delete_deployment(name)
    print(f"Deployment '{name}' deleted successfully.")


def deploy() -> object:
    """Deploy LLM server and print connection info."""
    print("=" * 60)
    print("Basilica LLM Server Deployment")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Deployment Name: {DEPLOYMENT_NAME}")
    print()
    print("Deploying to Basilica...")

    result = serve_llm()

    print()
    print("=" * 60)
    print("Deployment Ready")
    print("=" * 60)
    print(f"Deployment Name: {result.name}")
    print(f"Base URL: {result.url}")
    print(f"OpenAI API: {result.url}/v1")
    print()
    print("Use this URL with affinetes evaluation:")
    print(f"  --llm-url {result.url}/v1")
    print()
    print("Test with curl:")
    print(f'  curl {result.url}/v1/chat/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{MODEL_NAME}", "messages": [{{"role": "user", "content": "Hello"}}]}}\'')
    print()
    print("To delete this deployment when done:")
    print(f"  python basilica_llm_server.py --delete --name {result.name}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy or delete vLLM on Basilica")
    parser.add_argument("--delete", action="store_true", help="Delete deployment instead of creating")
    parser.add_argument("--name", type=str, help="Deployment name (for delete)")
    args = parser.parse_args()

    api_token = os.environ.get("BASILICA_API_TOKEN")
    if not api_token:
        print("Error: BASILICA_API_TOKEN environment variable not set")
        print("Please set: export BASILICA_API_TOKEN='your-token'")
        sys.exit(1)

    if args.delete:
        name = args.name or DEPLOYMENT_NAME
        delete_deployment(name)
    else:
        deploy()


if __name__ == "__main__":
    main()
