"""
Entry point for the PyInstaller bundle.
Usage: memora-sidecar --port PORT
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Memora AI sidecar")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import uvicorn
    from main import app

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
