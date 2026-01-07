from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import api, web  # Import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load resources
    yield
    # Shutdown: cleanup


app = FastAPI(
    title="Polylogue",
    description="Personal Knowledge Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api.router, prefix="/api")
app.include_router(web.router)

# Mount static files or frontend router later
