"""Serve the shared evaluation-service dashboard without loading model dependencies."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def add_status_ui(app: FastAPI, *, gpu_status: bool = False) -> None:
    """Add a root dashboard backed by the server's existing status endpoints."""
    assets = Path(__file__).with_name("status_assets")
    page = (assets / "index.html").read_text(encoding="utf-8")
    page = page.replace("__GPU_STATUS_ENABLED__", str(gpu_status).lower())

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def status_page() -> HTMLResponse:
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    app.mount("/status-assets", StaticFiles(directory=assets), name="status-assets")
