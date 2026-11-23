from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="A robust Retrieval-Augmented Generation (RAG) system with document ingestion, text chunking, vector search, and grounded LLM response generation.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Enable Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST API router
    app.include_router(api_router, prefix="/api/v1")

    # Static UI setup
    static_path = Path(__file__).parent / "static"
    static_path.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = static_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "message": f"Welcome to {settings.app_name} API. Visit /docs for interactive Swagger UI."
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
