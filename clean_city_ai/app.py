from __future__ import annotations

from urllib import request
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .ai import detect_waste, estimate_severity
from .database import assign_collector, get_all_complaints, get_user_by_email, init_db, save_complaint

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="CleanCity AI")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

init_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
):
    user = get_user_by_email(email)
    if user and user["password"] == password:
        if user["role"] == "admin":
            return RedirectResponse(url="/admin", status_code=303)
        if user["role"] == "collector":
            return RedirectResponse(url="/collector", status_code=303)
        return RedirectResponse(url="/citizen", status_code=303)
    return RedirectResponse(url="/", status_code=303)


@app.get("/citizen", response_class=HTMLResponse)
async def citizen_dashboard(request: Request):
    complaints = get_all_complaints()
    return templates.TemplateResponse(request=request, name="citizen.html", context={"complaints": complaints})


@app.get("/collector", response_class=HTMLResponse)
async def collector_dashboard(request: Request):
    complaints = get_all_complaints()
    return templates.TemplateResponse(request=request, name="collector.html", context={"complaints": complaints})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    complaints = get_all_complaints()
    pending = sum(1 for item in complaints if item["status"] == "Pending")
    in_progress = sum(1 for item in complaints if item["status"] == "In Progress")
    cleaned = sum(1 for item in complaints if item["status"] == "Cleaned")
    high_priority = sum(1 for item in complaints if item["severity"] in {"High", "Critical"})
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={           
            "complaints": complaints,
            "total": len(complaints),
            "pending": pending,
            "in_progress": in_progress,
            "cleaned": cleaned,
            "high_priority": high_priority,
        },
    )


@app.post("/report")
async def report(
    user_email: str = Form(...),
    waste_type: str = Form(""),
    description: str = Form(""),
    latitude: float = Form(...),
    longitude: float = Form(...),
):
    user = get_user_by_email(user_email)
    if not user:
        return RedirectResponse(url="/citizen", status_code=303)
    analysis = detect_waste(description or waste_type or "garbage on roadside", waste_type)
    severity = estimate_severity(description or waste_type or "garbage on roadside")
    assigned = assign_collector(latitude, longitude)
    payload = {
        "complaint_id": f"CC-{uuid.uuid4().hex[:8].upper()}",
        "user_id": user["id"],
        "image_url": "demo.png",
        "latitude": latitude,
        "longitude": longitude,
        "waste_type": analysis["waste_type"],
        "severity": severity["level"],
        "status": "Pending",
        "created_at": datetime.utcnow().isoformat(),
        "collector_id": assigned["id"] if assigned else None,
    }
    save_complaint(payload)
    return RedirectResponse(url="/citizen", status_code=303)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CleanCity AI"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("clean_city_ai.app:app", host="0.0.0.0", port=8000, reload=True)
