"""Vercel entrypoint — serves the guardrail gateway as a Python function.

Only the CPU-only gateway runs here. Set VLLM_BASE_URL (Vercel env var) to
your vLLM server so /v1/chat/completions has a model behind it; until then
it answers 502 backend_unavailable while /health and the input rails work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guardrails.gateway import app  # noqa: F401
