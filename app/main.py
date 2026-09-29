import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings, BASE_DIR, DATA_DIR, SNAPSHOTS_DIR
from app.core.logger import logger
from app.db.database import init_db
from app.api.routes import api_router
from app.services.video_stream import video_stream_manager
from app.services.attendance_manager import attendance_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database & Load Models into Cache
    logger.info("Initializing Local Face Attendance System...")
    init_db()
    attendance_manager.reload_cache()
    # Start video capture pipeline
    video_stream_manager.start()
    logger.info("Face Attendance System ready and listening.")
    yield
    # Shutdown
    logger.info("Shutting down Face Attendance System...")
    video_stream_manager.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS middleware for headless integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates
UI_DIR = BASE_DIR / "app" / "ui"
templates = Jinja2Templates(directory=str(UI_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")
app.mount("/data/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

# Include API Router
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def kiosk_page(request: Request):
    """Live Attendance Kiosk page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Admin and attendance management portal."""
    return templates.TemplateResponse(request=request, name="dashboard.html")
