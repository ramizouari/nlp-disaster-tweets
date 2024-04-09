"""Python script for running static testing."""

import subprocess
import sys

PATH = sys.executable.rsplit("/", maxsplit=1)[0]

print("[BLACK]", flush=True)
subprocess.run([f"{PATH}/python", "-m", "black", "--check", "."], check=True)

print("[FLAKE 8]", flush=True)
subprocess.run(
    [f"{PATH}/python", "-m", "flake8", "--config", "setup.ini", "."], check=True
)

print("[MYPY]", flush=True)
subprocess.run([f"{PATH}/python", "-m", "mypy", "."], check=True)
