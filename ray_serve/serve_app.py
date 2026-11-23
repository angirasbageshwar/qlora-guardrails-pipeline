"""Optional scale-out path: run the guardrail gateway as a Ray Serve deployment
so replicas autoscale with request load across a Ray cluster.

Start a cluster + deploy:
    ray start --head
    serve run ray_serve.serve_app:app        # gateway now behind Ray's proxy :8000/gw

vLLM keeps running as the model backend (start_vllm.sh); Ray scales the
stateless guardrail tier in front of it.
"""
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from ray import serve

from src.guardrails.gateway import apply_input_rails, apply_output_rails

fastapi_app = FastAPI()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 16,   # add a replica when each holds >16 in-flight
    },
)
@serve.ingress(fastapi_app)
class GuardrailGateway:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=120.0)

    @fastapi_app.get("/health")
    async def health(self) -> dict:
        return {"status": "ok", "tier": "ray-serve-gateway"}

    @fastapi_app.post("/v1/chat/completions")
    async def chat(self, request: Request):
        body = await request.json()
        body = apply_input_rails(body)          # raises 400 on jailbreak/PII-block
        resp = await self.client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions", json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        for choice in data.get("choices", []):
            content = choice.get("message", {}).get("content")
            if isinstance(content, str):
                clean, _ = apply_output_rails(content)
                choice["message"]["content"] = clean
        return data


app = GuardrailGateway.bind()
