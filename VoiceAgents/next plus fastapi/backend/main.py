from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from src.models.schemas import HealthResponse
from src.pipelines.sandwich import SandwichPipeline
from src.pipelines.realtime import RealtimePipeline

logging.basicConfig(level=logging.INFO)

app = FastAPI()

origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

sandwich_pipeline = SandwichPipeline(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_key=settings.azure_openai_api_key,
    azure_deployment=settings.azure_openai_deployment,
    elevenlabs_key=settings.elevenlabs_api_key,
    elevenlabs_voice_id=settings.elevenlabs_voice_id,
)

realtime_pipeline = RealtimePipeline(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_key=settings.azure_openai_api_key,
    azure_deployment=settings.azure_openai_realtime_deployment,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.websocket("/ws/sandwich")
async def ws_sandwich(websocket: WebSocket) -> None:
    await sandwich_pipeline.handle(websocket)


@app.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket) -> None:
    await realtime_pipeline.handle(websocket)
