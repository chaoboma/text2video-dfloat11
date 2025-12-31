import argparse
import sys
import os
from pathlib import Path
import traceback

# ANSI escape codes for colors (kept for stdout output in run_models)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

try:
    from .engine import generate_video
    from .hardware import get_available_models
    from . import db
    from . import migrations
    from .storage import record_generation
    from .paths import (
        ensure_initial_setup,
        get_data_dir,
        get_loras_dir,
        get_outputs_dir,
    )
    from .logger import get_logger, setup_logging
except ImportError:
    # Allow running as a script directly (e.g. python src/text2video/cli.py)
    sys.path.append(str(Path(__file__).parent))
    from engine import generate_video
    from hardware import get_available_models
    import db
    import migrations
    from storage import record_generation
    from paths import (
        ensure_initial_setup,
        get_data_dir,
        get_loras_dir,
        get_outputs_dir,
    )
    from logger import get_logger, setup_logging

# Directory Configuration
ensure_initial_setup()
OUTPUTS_DIR = get_outputs_dir()
LORAS_DIR = get_loras_dir()

logger = get_logger("text2video.cli")

def log_info(message: str):
    logger.info(message)

def log_warn(message: str):
    logger.warning(message)

def log_error(message: str):
    logger.error(message)

def run_models(args):
    models_response = get_available_models()
    
    # Print device info to stdout (CLI output)
    print(f"Device: {models_response['device'].upper()}")
    if models_response['ram_gb'] is not None:
        print(f"RAM: {models_response['ram_gb']:.1f} GB")
    if models_response['vram_gb'] is not None:
        print(f"VRAM: {models_response['vram_gb']:.1f} GB")

    print("\nAvailable Models:")
    if not models_response['models']:
        log_warn("No models available for this hardware configuration.")
        return

    for m in models_response['models']:
        rec_str = f" {GREEN}(Recommended){RESET}" if m.get('recommended') else ""
        print(f"  * {m['id']} -> {m['hf_model_id']}{rec_str}")

def run_list_loras(args):
    loras = db.list_loras()
    if not loras:
        print("No LoRAs found in database.")
        print("Use the web UI to upload LoRAs or place .safetensors files in the 'loras' folder and upload via API.")
        return

    print("Available LoRAs:")
    for l in loras:
        print(f"  * {l['display_name']} (File: {l['filename']}, ID: {l['id']})")

def run_server(args):
    """Start the FastAPI server via uvicorn."""
    import uvicorn

    # Handle both module execution and direct execution scenarios
    if __package__:
        from .network_utils import format_server_urls
    else:
        # When running directly (e.g., uv run src/text2video/cli.py serve)
        # Add the text2video directory to sys.path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        # Now network_utils.py should be directly importable
        import network_utils
        format_server_urls = network_utils.format_server_urls

    # Log paths first so they appear before MCP mount or uvicorn startup messages
    logger.info(f"Data Directory: {get_data_dir()}")
    logger.info(f"Outputs Directory: {get_outputs_dir()}")

    # Determine app string based on execution mode
    if not __package__:
        app_str = "server:app"
    else:
        app_str = "text2video.server:app"

    # Set environment variable to control MCP transport availability in web server
    if args.disable_mcp:
        os.environ["Zvideo_DISABLE_MCP"] = "1"
        log_info("    MCP: All web server endpoints disabled (/mcp and /mcp-sse)")

    # Display all accessible URLs
    server_urls = format_server_urls(args.host, args.port)
    log_info(f"Starting web server at:\n{server_urls}")

    uvicorn.run(
        app_str,
        host=args.host,
        port=args.port,
        reload=args.reload,
        timeout_graceful_shutdown=args.timeout_graceful_shutdown,
    )


def main():
    setup_logging()
    # Ensure DB is initialized
    migrations.init_db()

    parser = argparse.ArgumentParser(description="text2video using DFloat11/Wan2.2-T2V-A14B-DF11")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")


    # Subcommand: serve
    parser_serve = subparsers.add_parser("serve", help="Start Z-video Web Server")
    parser_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0 for all interfaces)")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
    parser_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser_serve.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=5,
        help="Seconds to wait for graceful shutdown before forcing exit (default: 5)",
    )

    # MCP transport options
    mcp_group = parser_serve.add_argument_group("MCP Transport Options")
    mcp_group.add_argument("--disable-mcp", action="store_true", help="Disable all MCP endpoints (/mcp and /mcp-sse)")

    parser_serve.set_defaults(func=run_server)

    # Subcommand: models
    parser_models = subparsers.add_parser("models", help="List available models and recommendations")
    parser_models.set_defaults(func=run_models)
    
    # Subcommand: loras
    parser_loras = subparsers.add_parser("loras", help="Manage LoRA models")
    loras_subparsers = parser_loras.add_subparsers(dest="subcommand", required=True)
    
    # loras list
    parser_loras_list = loras_subparsers.add_parser("list", help="List available LoRAs")
    parser_loras_list.set_defaults(func=run_list_loras)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
