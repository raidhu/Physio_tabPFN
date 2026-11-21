import os
from pathlib import Path
from typing import Optional


def load_env(env_path: Optional[Path] = None) -> None:
    """Simple .env loader that parses key=value without requiring external python-dotenv."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                os.environ[k] = v

    # Ensure TABPFN_TOKEN is set from api_key if present
    if "TABPFN_TOKEN" not in os.environ:
        if "api_key" in os.environ:
            os.environ["TABPFN_TOKEN"] = os.environ["api_key"]
        elif "API_KEY" in os.environ:
            os.environ["TABPFN_TOKEN"] = os.environ["API_KEY"]
