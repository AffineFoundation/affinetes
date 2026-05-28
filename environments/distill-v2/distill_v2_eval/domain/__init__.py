"""Domain layer: pure pydantic models, IDs, ports and domain errors.

No IO. No SQLAlchemy. No FastAPI. No httpx. ``application`` and adapters depend
on this; nothing in here depends on them.
"""
