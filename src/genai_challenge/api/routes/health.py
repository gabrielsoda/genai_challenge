from fastapi import APIRouter

router = APIRouter()

@router.get("/healthcheck")
async def healthchech() -> dict:
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}
