"""Docker image building from environment definitions"""

import docker
import importlib.util
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from ..utils.exceptions import ImageBuildError, ValidationError
from ..utils.logger import logger
from .env_detector import EnvDetector, EnvType


class ImageBuilder:
    """Builds Docker images from environment definitions"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            raise ImageBuildError(f"Failed to connect to Docker daemon: {e}")
    
    def build_from_env(
        self,
        env_path: str,
        image_tag: str,
        nocache: bool = False,
        quiet: bool = False,
        buildargs: Optional[dict] = None
    ) -> str:
        """
        Build Docker image from environment directory with automatic server injection
        
        Process:
        1. Detect environment type (function_based vs http_based)
        2. Build base image from user's Dockerfile
        3. If function_based: perform two-stage build to inject HTTP server
        4. Save environment metadata to image labels
        
        Expected directory structure:
            env_path/
                env.py          (required) - Main environment code
                Dockerfile      (required) - Dockerfile definition
                requirements.txt (optional) - Python dependencies
                *.py            (optional) - Additional Python modules
        
        Args:
            env_path: Path to environment directory
            image_tag: Image tag (e.g., "affine:latest")
            nocache: Don't use build cache
            quiet: Suppress build output
            buildargs: Docker build arguments (e.g., {"ENV_NAME": "webshop"})
            
        Returns:
            Built image tag
        """
        env_path = Path(env_path).resolve()
        
        # Validate environment directory
        if not env_path.is_dir():
            raise ValidationError(f"Environment path does not exist: {env_path}")
        
        env_file = env_path / "env.py"
        if not env_file.exists():
            raise ValidationError(
                f"Missing required env.py in {env_path}. "
                "Every environment must have an env.py file."
            )
        
        # Require Dockerfile
        dockerfile_path = env_path / "Dockerfile"
        if not dockerfile_path.exists():
            raise ValidationError(
                f"Missing required Dockerfile in {env_path}. "
                "Every environment must have a Dockerfile."
            )
        
        # Step 1: Detect environment type
        env_config = EnvDetector.detect(str(env_path))
        logger.info(f"Environment type: {env_config.env_type}")
        
        # Auto-resolve buildargs using config.py if exists
        config_path = env_path / "config.py"
        if config_path.exists() and buildargs:
            buildargs = self._resolve_buildargs(config_path, buildargs)
        
        # Step 2: Build base image
        base_image_tag = f"{image_tag}-base" if env_config.env_type == EnvType.FUNCTION_BASED else image_tag
        logger.info(f"Building base image '{base_image_tag}' from {env_path}")
        
        try:
            base_image_id = self._build_image(
                context_path=str(env_path),
                tag=base_image_tag,
                dockerfile="Dockerfile",
                buildargs=buildargs,
                nocache=nocache,
                quiet=quiet
            )
            logger.info(f"Base image built: {base_image_tag}")
            
        except Exception as e:
            raise ImageBuildError(f"Failed to build base image: {e}")
        
        # Step 3: Two-stage build for function_based environments
        if env_config.env_type == EnvType.FUNCTION_BASED:
            logger.info("Performing two-stage build to inject HTTP server...")
            final_image_id = self._inject_http_server(
                base_image_tag=base_image_tag,
                final_tag=image_tag,
                nocache=nocache,
                quiet=quiet
            )
        else:
            final_image_id = base_image_id
        
        # Step 4: Save metadata
        self._save_metadata(image_tag, env_config)
        
        logger.info(f"Successfully built image '{image_tag}'")
        return image_tag
    
    def _build_image(
        self,
        context_path: str,
        tag: str,
        dockerfile: str,
        buildargs: Optional[dict],
        nocache: bool,
        quiet: bool
    ) -> str:
        """
        Build Docker image and return image ID
        
        Args:
            context_path: Build context path
            tag: Image tag
            dockerfile: Dockerfile name
            buildargs: Build arguments
            nocache: Don't use cache
            quiet: Suppress output
            
        Returns:
            Image ID
        """
        try:
            if buildargs:
                logger.debug(f"Using build args: {buildargs}")
            
            # Stream build output in real-time
            build_logs = self.client.api.build(
                path=context_path,
                tag=tag,
                dockerfile=dockerfile,
                buildargs=buildargs or {},
                nocache=nocache,
                rm=True,
                decode=True
            )
            
            image_id = None
            for log in build_logs:
                if "stream" in log and not quiet:
                    print(log["stream"].rstrip(), flush=True)
                elif "error" in log:
                    print(log["error"].strip(), flush=True)
                    raise ImageBuildError(f"Build failed: {log['error']}")
                elif "aux" in log and "ID" in log["aux"]:
                    image_id = log["aux"]["ID"]
            
            if not image_id:
                raise ImageBuildError("Build completed but no image ID returned")
            
            return image_id
            
        except docker.errors.BuildError as e:
            error_msg = "Image build failed:\n"
            for log in e.build_log:
                if "error" in log:
                    error_msg += f"  {log['error']}\n"
                elif "stream" in log:
                    error_msg += f"  {log['stream']}"
            raise ImageBuildError(error_msg)
    
    def _inject_http_server(
        self,
        base_image_tag: str,
        final_tag: str,
        nocache: bool,
        quiet: bool
    ) -> str:
        """
        Inject HTTP server into base image using two-stage build
        
        Creates a temporary build context with:
        - http_wrapper.Dockerfile (FROM base_image)
        - http_server.py (server template)
        
        Args:
            base_image_tag: Base image tag
            final_tag: Final image tag
            nocache: Don't use cache
            quiet: Suppress output
            
        Returns:
            Final image ID
        """
        # Create temporary build context
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Get template directory
            template_dir = Path(__file__).parent.parent / "templates"

            # Copy HTTP server template
            server_template = template_dir / "http_server.py"
            shutil.copy2(server_template, tmpdir_path / "http_server.py")

            # Copy shared request_logger template
            request_logger_template = template_dir / "request_logger.py"
            if request_logger_template.exists():
                shutil.copy2(request_logger_template, tmpdir_path / "request_logger.py")
            
            # Generate Dockerfile with metadata label
            wrapper_dockerfile = template_dir / "http_wrapper.Dockerfile"
            dockerfile_content = wrapper_dockerfile.read_text()
            
            # Insert LABEL instruction before EXPOSE
            label_line = f'LABEL affinetes.env.type="{EnvType.FUNCTION_BASED}"\n'
            dockerfile_content = dockerfile_content.replace(
                "# Expose HTTP port\nEXPOSE 8000",
                f"# Save environment metadata\n{label_line}\n# Expose HTTP port\nEXPOSE 8000"
            )
            
            # Write modified Dockerfile
            (tmpdir_path / "Dockerfile").write_text(dockerfile_content)
            
            logger.debug(f"Two-stage build context created in {tmpdir}")
            
            # Build final image with base image as build arg
            image_id = self._build_image(
                context_path=tmpdir,
                tag=final_tag,
                dockerfile="Dockerfile",
                buildargs={"BASE_IMAGE": base_image_tag},
                nocache=nocache,
                quiet=quiet
            )
            
            return image_id
    
    def _save_metadata(self, image_tag: str, env_config) -> None:
        """
        Save environment metadata to image labels
        
        For http_based environments, we need to add labels after build
        since they provide their own Dockerfile without affinetes labels.
        
        Args:
            image_tag: Image tag
            env_config: Environment configuration
        """
        try:
            # For http_based environments, tag the image with metadata
            if env_config.env_type == EnvType.HTTP_BASED:
                image = self.client.images.get(image_tag)
                
                # Create a new Dockerfile to add label
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    
                    # Create minimal Dockerfile that just adds label
                    dockerfile_content = f"""FROM {image_tag}
LABEL affinetes.env.type="{env_config.env_type}"
"""
                    (tmpdir_path / "Dockerfile").write_text(dockerfile_content)
                    
                    # Build tagged version
                    temp_tag = f"{image_tag}-labeled"
                    self._build_image(
                        context_path=str(tmpdir_path),
                        tag=temp_tag,
                        dockerfile="Dockerfile",
                        buildargs=None,
                        nocache=True,
                        quiet=True
                    )
                    
                    # Remove old image and retag
                    self.client.images.remove(image_tag, force=True)
                    labeled_image = self.client.images.get(temp_tag)
                    labeled_image.tag(image_tag)
                    self.client.images.remove(temp_tag, force=True)
                    
                    logger.debug(f"Added metadata label to {image_tag} (type: {env_config.env_type})")
            else:
                # function_based images already have label from http_wrapper.Dockerfile
                image = self.client.images.get(image_tag)
                logger.debug(f"Image {image_tag} exists (type: {env_config.env_type})")
                
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def _resolve_buildargs(self, config_path: Path, buildargs: dict) -> dict:
        """
        Resolve build arguments using config.py
        
        Args:
            config_path: Path to config.py
            buildargs: Original build arguments
            
        Returns:
            Resolved build arguments (original merged with resolved config)
        """
        try:
            # Dynamically import config module
            spec = importlib.util.spec_from_file_location("env_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Look for standard resolve_buildargs function
            if hasattr(config_module, "resolve_buildargs"):
                resolve_func = getattr(config_module, "resolve_buildargs")
                logger.info(f"Found resolve_buildargs in {config_path}")
                
                resolved = resolve_func(buildargs)
                logger.info(f"Resolved build args: {resolved}")
                return resolved
            else:
                logger.debug(f"No resolve_buildargs function in {config_path}")
                return buildargs
                
        except Exception as e:
            logger.warning(f"Failed to resolve build args: {e}")
            return buildargs
    
    def push_image(self, image_tag: str, registry: Optional[str] = None) -> None:
        """
        Push image to registry
        
        Args:
            image_tag: Image tag to push
            registry: Registry URL (optional)
        """
        try:
            if registry:
                full_tag = f"{registry}/{image_tag}"
                logger.info(f"Tagging image {image_tag} as {full_tag}")
                image = self.client.images.get(image_tag)
                image.tag(full_tag)
                push_tag = full_tag
            else:
                push_tag = image_tag
            
            logger.info(f"Pushing image {push_tag}")
            
            for line in self.client.images.push(push_tag, stream=True, decode=True):
                if "status" in line:
                    logger.debug(f"{line['status']}")
                elif "error" in line:
                    raise ImageBuildError(f"Push failed: {line['error']}")
            
            logger.info(f"Successfully pushed {push_tag}")
            
        except docker.errors.APIError as e:
            raise ImageBuildError(f"Failed to push image: {e}")
        except Exception as e:
            raise ImageBuildError(f"Error pushing image: {e}")
    
    def pull_image(self, image_tag: str) -> str:
        """
        Pull image from registry
        
        Args:
            image_tag: Image tag to pull
            
        Returns:
            Image ID
        """
        try:
            logger.info(f"Pulling image {image_tag}")
            
            image = self.client.images.pull(image_tag)
            
            logger.info(f"Successfully pulled {image_tag} ({image.short_id})")
            return image.id
            
        except docker.errors.APIError as e:
            raise ImageBuildError(f"Failed to pull image: {e}")
        except Exception as e:
            raise ImageBuildError(f"Error pulling image: {e}")
    
    def image_exists(self, image_tag: str) -> bool:
        """
        Check if image exists locally
        
        Args:
            image_tag: Image tag to check
            
        Returns:
            True if image exists
        """
        try:
            self.client.images.get(image_tag)
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception:
            return False
    
    def remove_image(self, image_tag: str, force: bool = False) -> None:
        """
        Remove image
        
        Args:
            image_tag: Image tag to remove
            force: Force removal
        """
        try:
            logger.info(f"Removing image {image_tag}")
            self.client.images.remove(image_tag, force=force)
            logger.info(f"Image {image_tag} removed")
        except docker.errors.ImageNotFound:
            logger.warning(f"Image {image_tag} not found")
        except Exception as e:
            logger.error(f"Failed to remove image: {e}")