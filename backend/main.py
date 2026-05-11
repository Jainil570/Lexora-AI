"""FastAPI application entry point for Lexora backend."""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

from routers import health, documents, auth, analysis, chat
from utils.logger import set_database

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "lexora")

    logger.info(f"Connecting to MongoDB at {mongo_uri}...")
    app.state.mongo_client = AsyncIOMotorClient(mongo_uri)
    app.state.db = app.state.mongo_client[db_name]

    # Set database reference for audit logger
    set_database(app.state.db)

    # Verify connection
    try:
        await app.state.mongo_client.admin.command("ping")
        logger.info("MongoDB connection successful")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}. Audit logging will be disabled.")

    yield

    # Shutdown
    logger.info("Closing MongoDB connection...")
    app.state.mongo_client.close()


app = FastAPI(
    title="Lexora API",
    description="AI Legal Copilot for Indian Startups",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(analysis.router)
app.include_router(chat.router)
