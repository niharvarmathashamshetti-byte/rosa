"""
=============================================================================
ROSA Knee AI — Custom Exceptions & Error Handlers
=============================================================================
All application-specific errors are defined here.

These exceptions are caught by FastAPI's exception handlers and converted
into clean JSON responses — the user NEVER sees a raw Python traceback.

This is important for a medical imaging application where error messages
must be understandable by clinicians, not just developers.
=============================================================================
"""

from fastapi import Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Custom exception classes
# ---------------------------------------------------------------------------

class RosaBaseException(Exception):
    """Base exception for all ROSA Knee AI errors."""

    def __init__(self, message: str, status_code: int = 500, detail: str = ""):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class CaseNotFoundError(RosaBaseException):
    """Raised when a requested case ID does not exist."""

    def __init__(self, case_id: str):
        super().__init__(
            message=f"Case '{case_id}' not found.",
            status_code=404,
            detail="The requested case does not exist or has been deleted."
        )


class InvalidUploadError(RosaBaseException):
    """Raised when an uploaded file is invalid or unsupported."""

    def __init__(self, message: str = "Invalid upload."):
        super().__init__(
            message=message,
            status_code=400,
            detail="The uploaded file is not a supported medical image format."
        )


class ProcessingError(RosaBaseException):
    """Raised when preprocessing or other processing steps fail."""

    def __init__(self, message: str = "Processing failed."):
        super().__init__(
            message=message,
            status_code=500,
            detail="An error occurred during medical image processing."
        )


class ModelUnavailableError(RosaBaseException):
    """Raised when the segmentation model is not trained or loaded."""

    def __init__(self):
        super().__init__(
            message="Segmentation model is not available.",
            status_code=503,
            detail="The AI segmentation model has not been trained yet. "
                   "This is expected during development."
        )


class ValidationError(RosaBaseException):
    """Raised when CT validation fails."""

    def __init__(self, message: str = "CT validation failed."):
        super().__init__(
            message=message,
            status_code=422,
            detail="The uploaded medical image failed quality validation."
        )


# ---------------------------------------------------------------------------
# Exception handlers — register these with the FastAPI app
# ---------------------------------------------------------------------------

async def rosa_exception_handler(request: Request, exc: RosaBaseException) -> JSONResponse:
    """
    Converts any RosaBaseException into a clean JSON response.

    The user sees a structured error, never a Python stack trace.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "detail": exc.detail,
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches any unhandled exception and returns a safe generic message.

    This ensures that even unexpected errors don't leak internal details
    (file paths, database credentials, etc.) to the user.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An internal server error occurred.",
            "detail": "Please contact the development team if this persists."
        }
    )
