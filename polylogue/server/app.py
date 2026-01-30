from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from polylogue.version import VERSION_INFO

from . import api, web  # Import routes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: load resources
    yield
    # Shutdown: cleanup


app = FastAPI(
    title="Polylogue",
    description="Personal Knowledge Engine",
    version=VERSION_INFO.full,
    lifespan=lifespan,
)

app.include_router(api.router, prefix="/api")
app.include_router(web.router)

# Mount static files (served from templates dir)
templates_dir = Path(__file__).parent.parent / "templates"
app.mount("/static", StaticFiles(directory=str(templates_dir)), name="static")
