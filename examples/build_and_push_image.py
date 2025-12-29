"""
Build and Push All Images to Docker Registry

This script builds all environment images and pushes them to Docker Hub.
Useful for preparing images for distributed deployment across multiple machines.
"""

import time
import affinetes as af_env
from typing import Dict, List

# Registry configuration
REGISTRY = "docker.io/affinefoundation"

ENV_CONFIGS = {
    "game": {
        "path": "environments/openspiel",
        "image": "game:openspiel",
    },
    # "SWE-PRO": {
    #     "path": "environments/SWE-bench_Pro-os",
    #     "image": "swebench:pro",
    # },
    # "cde": {
    #     "path": "environments/primeintellect/cde",
    #     "image": "cde:pi",
    # },
    # "lgc": {
    #     "path": "environments/primeintellect/lgc",
    #     "image": "lgc:pi",
    # },
    # "mth": {
    #     "path": "environments/primeintellect/mth",
    #     "image": "mth:pi",
    # },
    # "sci": {
    #     "path": "environments/primeintellect/sci",
    #     "image": "sci:pi",
    # },
    # "affine": {
    #     "path": "environments/affine",
    #     "image": "affine-env:v4",
    # },
    # "agentgym:webshop": {
    #     "path": "environments/agentgym",
    #     "image": "agentgym:webshop",
    #     "buildargs": {"ENV_NAME": "webshop"},
    # },
    # "agentgym:alfworld": {
    #     "path": "environments/agentgym",
    #     "image": "agentgym:alfworld",
    #     "buildargs": {"ENV_NAME": "alfworld"},
    # },
    # "agentgym:babyai": {
    #     "path": "environments/agentgym",
    #     "image": "agentgym:babyai",
    #     "buildargs": {"ENV_NAME": "babyai"},
    # },
    # "agentgym:sciworld": {
    #     "path": "environments/agentgym",
    #     "image": "agentgym:sciworld",
    #     "buildargs": {"ENV_NAME": "sciworld"},
    # },
    # "agentgym:textcraft": {
    #     "path": "environments/agentgym",
    #     "image": "agentgym:textcraft",
    #     "buildargs": {"ENV_NAME": "textcraft"},
    # },
}


def build_and_push_images(registry: str = REGISTRY) -> Dict[str, str]:
    """
    Build and push all images to registry
    
    Args:
        registry: Docker registry URL (e.g., "docker.io/bignickeye")
        
    Returns:
        Dictionary mapping env_key to final image tag
    """
    print("\n" + "=" * 80)
    print(f"Building and Pushing Images to {registry}")
    print("=" * 80)
    
    results = {}
    built_images = set()
    total_start = time.time()
    
    for env_key, config in ENV_CONFIGS.items():
        image = config["image"]
        
        # Skip if already built
        if image in built_images:
            print(f"\n[SKIP] Image '{image}' already built")
            results[env_key] = f"{registry}/{image}"
            continue
        
        print(f"\n{'=' * 80}")
        print(f"[BUILD] Building '{image}' from '{config['path']}'...")
        print(f"{'=' * 80}")
        start = time.time()

        try:
            final_tag = af_env.build_image_from_env(
                env_path=config["path"],
                image_tag=image,
                buildargs=config.get("buildargs"),
                quiet=False,
                push=True,
                registry=registry
            )
            
            elapsed = time.time() - start
            print(f"\n[OK] Built and pushed '{final_tag}' in {elapsed:.1f}s")
            
            built_images.add(image)
            results[env_key] = final_tag
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n[ERROR] Failed to build/push '{image}' after {elapsed:.1f}s: {e}")
            raise
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("Build and Push Summary")
    print("=" * 80)
    print(f"Successfully built and pushed {len(built_images)} images in {total_elapsed:.1f}s")
    print(f"\nRegistry: {registry}")
    print(f"\nImages:")
    for env_key, final_tag in results.items():
        print(f"  - {env_key}: {final_tag}")
    print("=" * 80)
    
    return results


def verify_images(image_tags: List[str]) -> None:
    """
    Verify that images can be pulled from registry
    
    Args:
        image_tags: List of image tags to verify
    """
    print("\n" + "=" * 80)
    print("Verifying Images (Optional)")
    print("=" * 80)
    print("\nTo verify images can be pulled from remote machines, run:")
    print()
    for tag in image_tags:
        print(f"  docker pull {tag}")
    print()
    print("=" * 80)


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("Docker Image Build and Push Tool")
    print("=" * 80)
    print(f"\nRegistry: {REGISTRY}")
    print(f"Total images to build: {len(ENV_CONFIGS)}")
    print(f"Unique images: {len(set(c['image'] for c in ENV_CONFIGS.values()))}")
    
    # Confirm before proceeding
    print("\n⚠️  This will build and push images to Docker Hub")
    print("   Make sure you are logged in: docker login")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    try:
        # Build and push all images
        results = build_and_push_images(registry=REGISTRY)
        
        # Show verification commands
        verify_images(list(results.values()))
        
        print("\n✅ All images built and pushed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        raise


if __name__ == "__main__":
    main()
