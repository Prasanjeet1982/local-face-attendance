import uvicorn
from app.config import settings
from app.core.logger import logger

def start_server():
    logger.info(f"Starting Face Attendance API & Web Kiosk on http://{settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
