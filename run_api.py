"""
Run the Resume Optimizer API server.

Usage:
    python run_api.py
    
Or with custom settings:
    python run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run the Resume Optimizer API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use 1 for GPU)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Resume Optimizer API")
    print("=" * 60)
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {args.reload}")
    print(f"👷 Workers: {args.workers}")
    print("=" * 60)
    print(f"📖 API Docs: http://localhost:{args.port}/docs")
    print(f"📖 ReDoc: http://localhost:{args.port}/redoc")
    print("=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
