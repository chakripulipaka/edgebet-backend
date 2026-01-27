from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes import picks, simulation
from app.jobs.daily_job import scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - only start scheduler in production
    if settings.ENVIRONMENT == "production":
        scheduler.start()
    yield
    # Shutdown
    if settings.ENVIRONMENT == "production":
        scheduler.shutdown()


app = FastAPI(
    title="EdgeBet API",
    description="NBA betting analytics platform with ML-powered predictions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(picks.router, tags=["picks"])
app.include_router(simulation.router, tags=["simulation"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}
