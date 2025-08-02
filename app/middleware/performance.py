import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("loan_prediction.performance")

class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log performance metrics
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        # Add performance header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Alert on slow requests (>5 seconds)
        if process_time > 5.0:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} - "
                f"Time: {process_time:.4f}s"
            )
        
        return response