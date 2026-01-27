#!/usr/bin/env python3
"""Build OpenSpiel Docker image"""

import os
import affinetes as af

if __name__ == "__main__":
    # Change to affinetes root directory for relative path to work
    affinetes_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(affinetes_root)
    
    print("Building OpenSpiel image...")
    image_tag = af.build_image_from_env(
        env_path="environments/openspiel",
        image_tag="diagonalge/openspiel:latest"
    )
    print(f"✅ Image built successfully: {image_tag}")
