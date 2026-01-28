"""
FastAPI middleware for request tracking and logging.
"""

import time
import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

from genai_challenge.core.logging import logger

# Context variable to store request_id across async calls
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestLoggingMiddleware:
    """Pure ASGI middleware for request logging and tracking."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Process ASGI request."""
        # Only process HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # generate unique request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

        # extract request info from scope
        method = scope["method"]
        path = scope["path"]
        client = scope.get("client")
        client_host = client[0] if client else None

        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client": client_host,
            },
        )

        # Measure request duration
        start_time = time.time()

        # Track status code
        status_code = 200

        async def send_with_logging(message):
            """Wrapper to capture response status."""
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Add request_id to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        # Process request
        await self.app(scope, receive, send_with_logging)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log response
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()