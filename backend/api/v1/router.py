"""
Aggregates all v1 sub-routers into one APIRouter.
Mounted at /api/v1 in main.py.
"""
from fastapi import APIRouter
from backend.api.v1 import health, pipeline, documents, chat, config_api, guardrails

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(pipeline.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(config_api.router)
api_router.include_router(guardrails.router)
