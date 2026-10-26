"""Settings from environment variables and a local .env file (gitignored)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA = Path(os.getenv("DATA_DIR", "data"))


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env (see .env.example)")
    return value


def llm_api_key() -> str:
    return _required("DEEPSEEK_API_KEY")


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "deepseek-flash").strip()


def judge_model() -> str:
    """Drafts eval questions and grades answers: a stronger model than the one being evaluated."""
    return os.getenv("JUDGE_MODEL", "deepseek-v4-pro").strip()


def llm_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
