"""Main entry point for the FastAPI application."""

# CRITICAL: Apply compatibility fixes BEFORE any other imports
# Import torchvision fix first to register missing operators
import app.torchvision_fix  # noqa: F401, E402

# Apply torchaudio compatibility fix
try:
    import torchaudio
    if not hasattr(torchaudio, 'AudioMetaData'):
        from types import SimpleNamespace
        torchaudio.AudioMetaData = SimpleNamespace
except ImportError:
    pass

from collections.abc import AsyncGenerator

from app.core.warnings_filter import filter_warnings

filter_warnings()

import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, status  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.api import service_router, stt_router, task_router  # noqa: E402
from app.api.exception_handlers import (  # noqa: E402
    domain_error_handler,
    generic_error_handler,
    infrastructure_error_handler,
    task_not_found_handler,
    validation_error_handler,
)
from app.core.config import Config  # noqa: E402
from app.core.container import Container  # noqa: E402
from app.core.exceptions import (  # noqa: E402
    DomainError,
    InfrastructureError,
    TaskNotFoundError,
    ValidationError,
)
from app.docs import generate_db_schema, save_openapi_json  # noqa: E402
from app.infrastructure.database import Base, engine  # noqa: E402

# Load environment variables from .env
load_dotenv()

Base.metadata.create_all(bind=engine)

# Create dependency injection container
container = Container()

# Set container in dependencies module
from app.api import dependencies  # noqa: E402

dependencies.set_container(container)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for the FastAPI application.

    This function is used to perform startup and shutdown tasks for the FastAPI application.
    It saves the OpenAPI JSON and generates the database schema.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    logging.info("Application lifespan started - dependency container initialized")

    save_openapi_json(app)
    generate_db_schema(Base.metadata.tables.values())
    yield

    # Clean up container on shutdown
    logging.info("Shutting down application")


tags_metadata = [
    {
        "name": "Speech-2-Text",
        "description": "Operations related to transcript",
    },
    {
        "name": "Speech-2-Text services",
        "description": "Individual services for transcript",
    },
    {
        "name": "Tasks Management",
        "description": "Manage tasks.",
    },
    {
        "name": "Health",
        "description": "Health check endpoints to monitor application status",
    },
]


app = FastAPI(
    title="whisperX REST service",
    description=f"""
    # whisperX RESTful API

    Welcome to the whisperX RESTful API! This API provides a suite of audio processing services to enhance and analyze your audio content.

    ## Documentation:

    For detailed information on request and response formats, consult the [WhisperX Documentation](https://github.com/m-bain/whisperX).

    ## Services:

    Speech-2-Text provides a suite of audio processing services to enhance and analyze your audio content. The following services are available:

    1. Transcribe: Transcribe an audio/video  file into text.
    2. Align: Align the transcript to the audio/video file.
    3. Diarize: Diarize an audio/video file into speakers.
    4. Combine Transcript and Diarization: Combine the transcript and diarization results.

    ## Supported file extensions:
    AUDIO_EXTENSIONS = {Config.AUDIO_EXTENSIONS}

    VIDEO_EXTENSIONS = {Config.VIDEO_EXTENSIONS}

    """,
    version="0.0.1",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Register exception handlers
app.add_exception_handler(TaskNotFoundError, task_not_found_handler)
app.add_exception_handler(ValidationError, validation_error_handler)
app.add_exception_handler(DomainError, domain_error_handler)
app.add_exception_handler(InfrastructureError, infrastructure_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

# Include routers
app.include_router(stt_router)
app.include_router(task_router)
app.include_router(service_router)

# Mount static files for web interface
# Get the project root directory (parent of app directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_interface_path = os.path.join(project_root, "web_interface")

logging.info(f"Web interface path: {web_interface_path}")
logging.info(f"Web interface exists: {os.path.exists(web_interface_path)}")

if os.path.exists(web_interface_path):
    app.mount("/static", StaticFiles(directory=web_interface_path), name="static")
    logging.info("Static files mounted at /static")
else:
    logging.warning(f"Web interface directory not found at: {web_interface_path}")


@app.get("/", include_in_schema=False)
async def index():
    """Serve the web interface HTML page."""
    html_path = os.path.join(web_interface_path, "index.html")
    
    # Try multiple path resolutions
    possible_paths = [
        html_path,
        os.path.join(project_root, "web_interface", "index.html"),
        os.path.join(os.getcwd(), "web_interface", "index.html"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"Serving web interface HTML page from: {path}")
            return FileResponse(path, media_type="text/html")
    
    # If none of the paths exist, log and redirect to docs
    logging.warning(f"HTML file not found. Tried paths: {possible_paths}")
    logging.warning(f"Current working directory: {os.getcwd()}")
    logging.warning(f"Project root: {project_root}")
    logging.warning(f"Web interface path: {web_interface_path}")
    return RedirectResponse(url="/docs", status_code=307)




# Health check endpoints
@app.get("/health", tags=["Health"], summary="Simple health check")
async def health_check() -> JSONResponse:
    """Verify the service is up and running.

    Returns a simple status response to confirm the API service is operational.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "Service is running"},
    )


@app.get("/health/live", tags=["Health"], summary="Liveness check")
async def liveness_check() -> JSONResponse:
    """Check if the application is running.

    Used by orchestration systems like Kubernetes to detect if the app is alive.
    Returns timestamp along with status information.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "timestamp": time.time(),
            "message": "Application is live",
        },
    )


@app.get("/health/ready", tags=["Health"], summary="Readiness check")
async def readiness_check() -> JSONResponse:
    """Check if the application is ready to accept requests.

    Verifies dependencies like the database are connected and ready.
    Returns HTTP 200 if all systems are operational, HTTP 503 if any dependency
    has failed.
    """
    try:
        # Check database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "database": "connected",
                "message": "Application is ready to accept requests",
            },
        )
    except Exception:
        logging.exception("Readiness check failed:")

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "database": "disconnected",
                "message": "Application is not ready due to an internal error.",
            },
        )
