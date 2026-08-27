from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .ai import (
    compute_priority_score,
    detect_waste_from_image,
    estimate_severity,
    predict_hotspots,
    verify_cleanup,
)
from .database import (
    assign_collector,
    find_duplicate_complaint,
    get_admin_stats,
    get_all_complaints,
    get_collector_by_user_email,
    get_complaint_by_id,
    get_complaints_for_citizen,
    get_complaints_for_collector,
    get_hotspot_clusters,
    get_user_by_email,
    increment_duplicate_report,
    init_db,
    save_complaint,
    save_resolution,
    update_complaint_status,
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CleanCity AI", description="Smart Garbage Reporting & Collection System")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

init_db()


def _save_upload(file: UploadFile | None) -> str | None:
    if not file or not file.filename:
        return None
    ext = Path(file.filename).suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return f"/static/uploads/{filename}"


def _session_user(request: Request) -> dict | None:
    email = request.cookies.get("user_email")
    if not email:
        return None
    return get_user_by_email(email)


def _require_role(request: Request, role: str) -> dict | None:
    user = _session_user(request)
    if not user or user["role"] != role:
        return None
    return user


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or user["password"] != password:
        return RedirectResponse(url="/?error=invalid", status_code=303)

    redirect_map = {
        "admin": "/admin",
        "collector": "/collector",
        "citizen": "/citizen",
    }
    response = RedirectResponse(url=redirect_map.get(user["role"], "/citizen"), status_code=303)
    response.set_cookie(key="user_email", value=user["email"], httponly=True, max_age=86400)
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("user_email")
    return response


@app.get("/citizen", response_class=HTMLResponse)
async def citizen_dashboard(request: Request):
    user = _require_role(request, "citizen")
    if not user:
        return RedirectResponse(url="/", status_code=303)
    complaints = get_complaints_for_citizen(user["id"])
    return templates.TemplateResponse(
        request=request,
        name="citizen.html",
        context={"complaints": complaints, "user": user},
    )


@app.get("/collector", response_class=HTMLResponse)
async def collector_dashboard(request: Request):
    user = _require_role(request, "collector")
    if not user:
        return RedirectResponse(url="/", status_code=303)
    collector = get_collector_by_user_email(user["email"])
    complaints = get_complaints_for_collector(collector["id"]) if collector else []
    return templates.TemplateResponse(
        request=request,
        name="collector.html",
        context={"complaints": complaints, "user": user, "collector": collector},
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    user = _require_role(request, "admin")
    if not user:
        return RedirectResponse(url="/", status_code=303)
        
    complaints = get_all_complaints()
    hotspots = get_hotspot_clusters()
    stats = get_admin_stats()
    predictions = predict_hotspots(
        [{"latitude": c["latitude"], "longitude": c["longitude"]} for c in complaints]
    )
    
    complaints_json = json.dumps(complaints, default=str)
    hotspots_json = json.dumps(hotspots, default=str)

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "complaints": complaints,
            "hotspots": hotspots,
            "predictions": predictions,
            "complaints_json": complaints_json,
            "hotspots_json": hotspots_json,
            "user": user,
            **stats,
        },
    )


@app.post("/report")
async def report(
    request: Request,
    waste_type: str = Form(""),
    description: str = Form(""),
    latitude: float = Form(...),
    longitude: float = Form(...),
    photo: UploadFile | None = File(None),
):
    user = _require_role(request, "citizen")
    if not user:
        return RedirectResponse(url="/", status_code=303)

    if not (waste_type or "").strip() and not (description or "").strip() and (not photo or not photo.filename):
        return RedirectResponse(url="/citizen?msg=incomplete", status_code=303)

    image_url = _save_upload(photo)
    local_path = BASE_DIR / image_url.lstrip("/") if image_url else None

    analysis = detect_waste_from_image(local_path, description, waste_type)
    if not analysis.get("is_garbage", True):
        return RedirectResponse(url="/citizen?msg=no_garbage", status_code=303)

    severity = analysis.get("severity") or estimate_severity(description or waste_type or "garbage")
    duplicate = find_duplicate_complaint(latitude, longitude)

    if duplicate:
        updated = increment_duplicate_report(duplicate["complaint_id"])
        priority = compute_priority_score(
            {"Low": 0.3, "Medium": 0.55, "High": 0.85, "Critical": 0.95}.get(updated["severity"], 0.5),
            updated.get("report_count", 1),
        )
        update_complaint_status(duplicate["complaint_id"], updated["status"], priority_score=priority)
        return RedirectResponse(url="/citizen?msg=duplicate", status_code=303)

    assigned = assign_collector(latitude, longitude)
    status = "Assigned" if assigned else "Pending"
    priority = compute_priority_score(severity["score"], 1)

    payload = {
        "complaint_id": f"CC-{uuid.uuid4().hex[:8].upper()}",
        "user_id": user["id"],
        "image_url": image_url or "",
        "before_image": image_url or "",
        "latitude": latitude,
        "longitude": longitude,
        "waste_type": analysis["waste_type"],
        "severity": severity["level"],
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
        "collector_id": assigned["id"] if assigned else None,
        "ai_confidence": analysis.get("confidence"),
        "description": description,
        "priority_score": priority,
    }
    save_complaint(payload)
    return RedirectResponse(url="/citizen?msg=reported", status_code=303)


@app.post("/collector/{complaint_id}/accept")
async def accept_task(request: Request, complaint_id: str):
    user = _require_role(request, "collector")
    if not user:
        return RedirectResponse(url="/", status_code=303)
    update_complaint_status(complaint_id, "Assigned")
    return RedirectResponse(url="/collector?msg=accepted", status_code=303)


@app.post("/collector/{complaint_id}/start")
async def start_cleaning(request: Request, complaint_id: str):
    user = _require_role(request, "collector")
    if not user:
        return RedirectResponse(url="/", status_code=303)
    update_complaint_status(complaint_id, "In Progress")
    return RedirectResponse(url="/collector?msg=started", status_code=303)


@app.post("/collector/{complaint_id}/complete")
async def complete_task(
    request: Request,
    complaint_id: str,
    after_photo: UploadFile | None = File(None),
):
    user = _require_role(request, "collector")
    if not user:
        return RedirectResponse(url="/", status_code=303)

    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        return RedirectResponse(url="/collector", status_code=303)

    after_url = _save_upload(after_photo)
    before_path = BASE_DIR / complaint["before_image"].lstrip("/") if complaint.get("before_image") else None
    after_path = BASE_DIR / after_url.lstrip("/") if after_url else None

    verification = verify_cleanup(before_path, after_path)
    verified_flag = 1 if verification["verified"] else 0
    new_status = "Verified" if verification["verified"] else "Cleaned"

    update_complaint_status(
        complaint_id,
        new_status,
        after_image=after_url or "",
        verified=verified_flag,
    )
    save_resolution(
        {
            "complaint_id": complaint_id,
            "before_image": complaint.get("before_image"),
            "after_image": after_url,
            "completed_time": datetime.utcnow().isoformat(),
            "verified": verified_flag,
            "verification_score": verification.get("score"),
        }
    )
    return RedirectResponse(url="/collector?msg=completed", status_code=303)


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "CleanCity AI", "version": "1.0.0"}


@app.get("/api/complaints")
async def api_complaints():
    return get_all_complaints()


@app.get("/api/complaints/{complaint_id}")
async def api_complaint_detail(complaint_id: str):
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return complaint


@app.get("/api/hotspots")
async def api_hotspots():
    complaints = get_all_complaints()
    return {
        "clusters": get_hotspot_clusters(),
        "predictions": predict_hotspots(
            [{"latitude": c["latitude"], "longitude": c["longitude"]} for c in complaints]
        ),
    }


@app.get("/api/stats")
async def api_stats():
    return get_admin_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("clean_city_ai.app:app", host="127.0.0.1", port=8000, reload=True)
