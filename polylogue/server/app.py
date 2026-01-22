from contextlib import asynccontextmanager

from fastapi import FastAPI

from polylogue.version import VERSION_INFO

from . import api, web  # Import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Mount static files or frontend router later
