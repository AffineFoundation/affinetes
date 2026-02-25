"""
Build LiveWeb Arena image directly from its Git repository URL.

Demonstrates the URL-based build feature: no need to clone the repo manually.
Supports Docker-style fragment syntax for branch/subfolder selection:

    https://github.com/org/repo.git#branch:path/to/env

Usage:
    python examples/liveweb/build_from_repo.py
    python examples/liveweb/build_from_repo.py --tag liveweb-arena:dev
    python examples/liveweb/build_from_repo.py --push --registry docker.io/myuser
"""

import argparse
import affinetes as af_env


LIVEWEB_REPO_URL = "https://github.com/AffineFoundation/liveweb-arena.git"


def main():
    parser = argparse.ArgumentParser(description="Build LiveWeb Arena from repo URL")
    parser.add_argument("--tag", default="liveweb-arena:latest", help="Image tag")
    parser.add_argument("--push", action="store_true", help="Push to registry after build")
    parser.add_argument("--registry", default=None, help="Registry URL (e.g. docker.io/myuser)")
    parser.add_argument("--nocache", action="store_true", help="Build without Docker cache")
    args = parser.parse_args()

    print(f"Building '{args.tag}' from {LIVEWEB_REPO_URL}")

    tag = af_env.build_image_from_env(
        env_path=LIVEWEB_REPO_URL,
        image_tag=args.tag,
        nocache=args.nocache,
        push=args.push,
        registry=args.registry,
    )

    print(f"Built: {tag}")


if __name__ == "__main__":
    main()
