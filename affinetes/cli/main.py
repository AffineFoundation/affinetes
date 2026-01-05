"""Main CLI entry point for affinetes"""

import sys
import argparse
import asyncio
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from ..utils.logger import logger
from .commands import run_environment, call_method, build_and_push, init_environment, test_environment

load_dotenv(override=True)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI"""
    
    parser = argparse.ArgumentParser(
        prog='afs',
        description='Affinetes CLI - Container-based Environment Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start environment from image
  afs run bignickeye/affine:v2 --env CHUTES_API_KEY=xxx
  
  # Start from directory (auto build)
  afs run --dir environments/affine --tag affine:v2
  
  # Call method
  afs call affine-v2 evaluate --arg task_type=abd --arg num_samples=2
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # === run command ===
    run_parser = subparsers.add_parser(
        'run',
        help='Start an environment container'
    )
    run_parser.add_argument(
        'image',
        nargs='?',
        help='Docker image name (e.g., bignickeye/affine:v2)'
    )
    run_parser.add_argument(
        '--dir',
        dest='env_dir',
        help='Build from environment directory'
    )
    run_parser.add_argument(
        '--tag',
        help='Image tag when building from directory (default: auto-generated)'
    )
    run_parser.add_argument(
        '--name',
        help='Container name (default: derived from image)'
    )
    run_parser.add_argument(
        '--env',
        action='append',
        dest='env_vars',
        help='Environment variable (format: KEY=VALUE, can be specified multiple times)'
    )
    run_parser.add_argument(
        '--pull',
        action='store_true',
        help='Pull image before starting'
    )
    run_parser.add_argument(
        '--mem-limit',
        help='Memory limit (e.g., 512m, 1g, 2g)'
    )
    run_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cache when building (only with --dir)'
    )
    
    # === build command ===
    build_parser = subparsers.add_parser(
        'build',
        help='Build and optionally push environment image'
    )
    build_parser.add_argument(
        'env_dir',
        help='Environment directory path'
    )
    build_parser.add_argument(
        '--tag',
        required=True,
        help='Image tag (e.g., myimage:v1 or registry.io/myimage:v1)'
    )
    build_parser.add_argument(
        '--push',
        action='store_true',
        help='Push image to registry after build'
    )
    build_parser.add_argument(
        '--registry',
        help='Registry URL (e.g., docker.io/username, ghcr.io/org)'
    )
    build_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cache when building'
    )
    build_parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress build output'
    )
    build_parser.add_argument(
        '--build-arg',
        action='append',
        dest='build_args',
        help='Docker build arguments (format: KEY=VALUE, can be specified multiple times)'
    )
    
    # === init command ===
    init_parser = subparsers.add_parser(
        'init',
        help='Initialize a new environment directory'
    )
    init_parser.add_argument(
        'name',
        help='Environment name (will create directory with this name)'
    )
    init_parser.add_argument(
        '--type',
        choices=['function', 'http'],
        default='function',
        help='Environment type: function (default) or http'
    )
    init_parser.add_argument(
        '--template',
        choices=['basic', 'actor', 'fastapi'],
        default='basic',
        help='Template type: basic (module functions), actor (Actor class), or fastapi (HTTP-based)'
    )
    
    # === call command ===
    call_parser = subparsers.add_parser(
        'call',
        help='Call a method on running environment'
    )
    call_parser.add_argument(
        'name',
        help='Environment/container name'
    )
    call_parser.add_argument(
        'method',
        help='Method name to call'
    )
    call_parser.add_argument(
        '--arg',
        action='append',
        dest='args',
        help='Method argument (format: KEY=VALUE, can be specified multiple times)'
    )
    call_parser.add_argument(
        '--json',
        dest='json_args',
        help='JSON string for complex arguments'
    )
    call_parser.add_argument(
        '--timeout',
        type=int,
        help='Timeout in seconds'
    )

    # === validate command ===
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate environment seed consistency and generate rollouts'
    )
    validate_parser.add_argument(
        'env_dir',
        help='Environment directory path'
    )
    validate_parser.add_argument(
        '--num-tests',
        type=int,
        default=100,
        help='Number of tests to run (default: 100)'
    )
    validate_parser.add_argument(
        '--task-id-start',
        type=int,
        default=1,
        help='Starting task_id (default: 1)'
    )
    validate_parser.add_argument(
        '--task-id-end',
        type=int,
        help='Ending task_id (default: start + num_tests - 1)'
    )
    validate_parser.add_argument(
        '--output',
        default='rollouts',
        help='Output directory for rollouts (default: rollouts/)'
    )
    validate_parser.add_argument(
        '--api-key',
        help='API key for LLM service (default: CHUTES_API_TOKEN env var)'
    )
    validate_parser.add_argument(
        '--base-url',
        help='Base URL for LLM API (default: auto-detect from MINER_SLUG)'
    )
    validate_parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for LLM generation (default: 0.7)'
    )
    validate_parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout for each evaluation in seconds (default: 60)'
    )

    return parser


def parse_env_vars(env_list: Optional[list]) -> Dict[str, str]:
    """Parse environment variables from KEY=VALUE format"""
    env_vars = {}
    if env_list:
        for env_str in env_list:
            if '=' not in env_str:
                logger.warning(f"Invalid env var format (should be KEY=VALUE): {env_str}")
                continue
            key, value = env_str.split('=', 1)
            env_vars[key] = value
    return env_vars


def parse_method_args(args_list: Optional[list], json_str: Optional[str]) -> Dict[str, Any]:
    """Parse method arguments from --arg and --json"""
    method_args = {}
    
    # Parse --arg KEY=VALUE
    if args_list:
        for arg_str in args_list:
            if '=' not in arg_str:
                logger.warning(f"Invalid arg format (should be KEY=VALUE): {arg_str}")
                continue
            key, value = arg_str.split('=', 1)
            
            # Try to parse as JSON value for complex types
            try:
                method_args[key] = json.loads(value)
            except json.JSONDecodeError:
                # Keep as string if not valid JSON
                method_args[key] = value
    
    # Parse --json (overrides --arg)
    if json_str:
        try:
            json_args = json.loads(json_str)
            method_args.update(json_args)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            sys.exit(1)
    
    return method_args


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        # Route to command handlers
        if args.command == 'run':
            # Validate run arguments
            if not args.image and not args.env_dir:
                parser.error("Either IMAGE or --dir must be specified")
            
            env_vars = parse_env_vars(args.env_vars)
            
            asyncio.run(run_environment(
                image=args.image,
                env_dir=args.env_dir,
                tag=args.tag,
                name=args.name,
                env_vars=env_vars,
                pull=args.pull,
                mem_limit=args.mem_limit,
                no_cache=args.no_cache
            ))
        
        elif args.command == 'build':
            build_args = parse_env_vars(args.build_args)
            
            asyncio.run(build_and_push(
                env_dir=args.env_dir,
                tag=args.tag,
                push=args.push,
                registry=args.registry,
                no_cache=args.no_cache,
                quiet=args.quiet,
                build_args=build_args
            ))
        
        elif args.command == 'init':
            init_environment(
                name=args.name,
                env_type=args.type,
                template=args.template
            )
        
        elif args.command == 'call':
            method_args = parse_method_args(args.args, args.json_args)

            asyncio.run(call_method(
                name=args.name,
                method=args.method,
                args=method_args,
                timeout=args.timeout
            ))

        elif args.command == 'validate':
            asyncio.run(test_environment(
                env_dir=args.env_dir,
                num_tests=args.num_tests,
                task_id_start=args.task_id_start,
                task_id_end=args.task_id_end,
                output_dir=args.output,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                timeout=args.timeout
            ))

        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()