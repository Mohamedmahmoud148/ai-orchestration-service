from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import health, chat

# Initialize logging
setup_logging()

app = FastAPI(
    title="AI Support Service",
    description="FastAPI service orchestrating Gemini intents to .NET backend execution.",
    version="1.0.0"
)

# Allow CORS for frontends (configure specifically for production env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router)
app.include_router(chat.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI AI Service. See /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the application gracefully
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=settings.PORT, 
        reload=(settings.ENVIRONMENT == "development")
    )
