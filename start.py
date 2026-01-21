"""
IMPULATOR Startup Script.

Starts both backend (FastAPI) and frontend (Streamlit) properly.
Run with: python start.py
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

# Configuration
BACKEND_HOST = os.getenv("API_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("API_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

PROJECT_ROOT = Path(__file__).parent
processes = []


def start_backend():
    """Start FastAPI backend."""
    print(f"[BACKEND] Starting on http://{BACKEND_HOST}:{BACKEND_PORT}")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "--reload" if os.getenv("DEBUG") else "--no-access-log",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(("backend", process))
    return process


def start_frontend():
    """Start Streamlit frontend."""
    print(f"[FRONTEND] Starting on http://localhost:{FRONTEND_PORT}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", str(FRONTEND_PORT),
        "--server.headless", "true",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(("frontend", process))
    return process


def wait_for_backend(timeout=30):
    """Wait for backend to be ready."""
    import urllib.request
    import urllib.error

    url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/v1/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print("[BACKEND] Ready!")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)

    print("[BACKEND] Failed to start within timeout")
    return False


def cleanup(signum=None, frame=None):
    """Clean up all processes."""
    print("\n[SHUTDOWN] Stopping all services...")

    for name, process in processes:
        if process.poll() is None:
            print(f"[SHUTDOWN] Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    print("[SHUTDOWN] Done")
    sys.exit(0)


def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 50)
    print("IMPULATOR - Starting Services")
    print("=" * 50)

    # Start backend first
    backend = start_backend()

    # Wait for backend to be ready
    if not wait_for_backend():
        print("[ERROR] Backend failed to start. Check logs above.")
        cleanup()
        return

    # Start frontend
    frontend = start_frontend()

    print("=" * 50)
    print(f"Backend:  http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print(f"Frontend: http://localhost:{FRONTEND_PORT}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    # Wait for processes
    try:
        while True:
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"[ERROR] {name} exited with code {process.returncode}")
                    cleanup()
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
