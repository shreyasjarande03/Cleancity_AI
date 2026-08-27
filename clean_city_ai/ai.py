from __future__ import annotations

from pathlib import Path
from typing import Any

WASTE_KEYWORDS = {
    "Plastic": ["plastic", "bottle", "packaging", "wrapper", "bag", "container", "cup"],
    "Organic": ["food", "fruit", "vegetable", "organic", "garden", "leaf", "compost", "banana"],
    "Mixed": ["mixed", "trash", "garbage", "waste", "pile", "dump", "litter"],
    "Construction": ["brick", "cement", "construction", "debris", "concrete", "tile", "rubble"],
    "Dry": ["paper", "cardboard", "dry", "metal", "glass", "carton", "newspaper"],
}

# COCO class names mapped to waste categories for YOLO fallback
COCO_WASTE_MAP = {
    "bottle": "Plastic",
    "cup": "Plastic",
    "bowl": "Plastic",
    "banana": "Organic",
    "apple": "Organic",
    "orange": "Organic",
    "broccoli": "Organic",
    "carrot": "Organic",
    "sandwich": "Organic",
    "pizza": "Organic",
    "donut": "Organic",
    "cake": "Organic",
    "book": "Dry",
    "scissors": "Dry",
    "cell phone": "Dry",
    "remote": "Dry",
    "keyboard": "Dry",
    "laptop": "Dry",
    "tv": "Dry",
    "microwave": "Dry",
    "refrigerator": "Dry",
}

_yolo_model = None
_yolo_load_attempted = False


def _load_yolo():
    global _yolo_model, _yolo_load_attempted
    if _yolo_load_attempted:
        return _yolo_model
    _yolo_load_attempted = True
    try:
        from ultralytics import YOLO

        weights = Path(__file__).resolve().parent.parent / "yolov8n.pt"
        _yolo_model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")
    except Exception:
        _yolo_model = None
    return _yolo_model


def _analyze_image_pixels(image_path: str | Path) -> dict[str, Any]:
    try:
        from PIL import Image
        import statistics

        img = Image.open(image_path).convert("RGB")
        pixels = list(img.getdata())
        w, h = img.size
        total = w * h

        brightness = statistics.mean(sum(p) / 3 for p in pixels)
        green_ratio = sum(1 for r, g, b in pixels if g > r and g > b) / total
        brown_ratio = sum(1 for r, g, b in pixels if r > 80 and g > 50 and b < 80) / total
        gray_ratio = sum(1 for r, g, b in pixels if abs(r - g) < 20 and abs(g - b) < 20) / total

        clutter_score = min(1.0, (gray_ratio * 1.5) + (brown_ratio * 1.2) + 0.1)

        waste_type = "Mixed"
        if green_ratio > 0.25:
            waste_type = "Organic"
        elif gray_ratio > 0.35:
            waste_type = "Mixed"
        elif brown_ratio > 0.2:
            waste_type = "Construction"

        return {
            "clutter_score": clutter_score,
            "waste_type_hint": waste_type,
            "brightness": brightness / 255,
        }
    except Exception:
        return {"clutter_score": 0.5, "waste_type_hint": "Mixed", "brightness": 0.5}


def detect_waste_from_image(
    image_path: str | Path | None,
    image_description: str = "",
    user_category: str | None = None,
) -> dict[str, Any]:
    """Detect garbage using YOLO when available, with pixel and keyword fallbacks."""
    text = (image_description or "").lower()

    if user_category and user_category.strip().title() in WASTE_KEYWORDS and not image_path:
        return {"is_garbage": True, "waste_type": user_category.strip().title(), "confidence": 0.85, "method": "user"}

    if not image_path:
        return detect_waste(text, user_category)

    path = Path(image_path)
    if not path.exists():
        return detect_waste(text, user_category)

    model = _load_yolo()
    pixel_info = _analyze_image_pixels(path)

    if model is not None:
        try:
            results = model(str(path), verbose=False)
            detections = []
            total_area = 0.0

            for result in results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = names.get(cls_id, "unknown").lower()
                    xyxy = box.xyxy[0].tolist()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    total_area += area
                    detections.append({"class": name, "confidence": conf, "area": area})

            if detections:
                waste_hits = [d for d in detections if d["class"] in COCO_WASTE_MAP]
                if waste_hits:
                    best = max(waste_hits, key=lambda d: d["confidence"] * d["area"])
                    waste_type = COCO_WASTE_MAP[best["class"]]
                    confidence = min(0.98, best["confidence"] + 0.05)
                else:
                    waste_type = pixel_info["waste_type_hint"]
                    confidence = 0.65 + pixel_info["clutter_score"] * 0.2

                img_w, img_h = _get_image_size(path)
                area_ratio = total_area / max(1, img_w * img_h)
                severity = estimate_severity_from_signals(area_ratio, pixel_info["clutter_score"], text)

                return {
                    "is_garbage": True,
                    "waste_type": user_category.strip().title() if user_category and user_category.strip() else waste_type,
                    "confidence": round(confidence, 2),
                    "method": "yolo",
                    "detections": len(detections),
                    "severity": severity,
                }
        except Exception:
            pass

    clutter = pixel_info["clutter_score"]
    if clutter < 0.15 and ("clean" in text or "no garbage" in text):
        return {"is_garbage": False, "waste_type": "Unknown", "confidence": 0.88, "method": "pixels"}

    keyword_result = detect_waste(text or "garbage waste", user_category)
    waste_type = keyword_result["waste_type"]
    if waste_type == "Unknown" or waste_type == "Mixed":
        waste_type = pixel_info["waste_type_hint"]

    severity = estimate_severity_from_signals(clutter * 0.4, clutter, text)
    return {
        "is_garbage": True,
        "waste_type": user_category.strip().title() if user_category and user_category.strip() else waste_type,
        "confidence": round(0.6 + clutter * 0.3, 2),
        "method": "pixels+keywords",
        "severity": severity,
    }


def _get_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        return Image.open(path).size
    except Exception:
        return (640, 480)


def detect_waste(image_description: str, user_category: str | None = None) -> dict[str, Any]:
    text = (image_description or "").lower()
    selected = user_category.strip().title() if user_category and user_category.strip() else None
    if selected in WASTE_KEYWORDS:
        return {"is_garbage": True, "waste_type": selected, "confidence": 0.96, "method": "user"}

    if (
        selected == "None"
        or text.strip() == "none"
        or "no garbage" in text
        or "no litter" in text
        or "clean road" in text
        or "clean scene" in text
    ):
        return {"is_garbage": False, "waste_type": "Unknown", "confidence": 0.92, "method": "keywords"}

    for waste_type, keywords in WASTE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return {"is_garbage": True, "waste_type": waste_type, "confidence": 0.9, "method": "keywords"}

    return {"is_garbage": True, "waste_type": "Mixed", "confidence": 0.72, "method": "keywords"}


def estimate_severity(description: str) -> dict[str, Any]:
    return estimate_severity_from_signals(0.0, 0.0, description)


def estimate_severity_from_signals(area_ratio: float, clutter_score: float, description: str) -> dict[str, Any]:
    text = (description or "").lower()
    score = max(area_ratio, clutter_score * 0.5)

    if any(kw in text for kw in ["critical", "drainage", "block", "overflow"]):
        level = "Critical"
        score = max(score, 0.95)
    elif score > 0.35 or any(kw in text for kw in ["large", "pile", "dump", "big", "heap"]):
        level = "High"
        score = max(score, 0.85)
    elif score > 0.15 or any(kw in text for kw in ["medium", "several", "roadside"]):
        level = "Medium"
        score = max(score, 0.55)
    else:
        level = "Low"
        score = max(score, 0.3)

    return {"level": level, "score": round(min(1.0, score), 2)}


def compute_priority_score(severity_score: float, report_count: int, age_hours: float = 0) -> float:
    duplicate_boost = min(0.3, report_count * 0.05)
    age_boost = min(0.2, age_hours * 0.01)
    return round(min(1.0, severity_score + duplicate_boost + age_boost), 2)


def verify_cleanup(before_path: str | Path | None, after_path: str | Path | None) -> dict[str, Any]:
    """Compare before/after images to verify garbage removal."""
    if not before_path or not after_path:
        return {"verified": False, "score": 0.0, "reason": "Missing images"}

    before = Path(before_path)
    after = Path(after_path)
    if not before.exists() or not after.exists():
        return {"verified": False, "score": 0.0, "reason": "Image files not found"}

    try:
        from PIL import Image
        import statistics

        b_img = Image.open(before).convert("RGB").resize((256, 256))
        a_img = Image.open(after).convert("RGB").resize((256, 256))

        b_pixels = list(b_img.getdata())
        a_pixels = list(a_img.getdata())

        diffs = [abs(sum(b) - sum(a)) / 765 for b, a in zip(b_pixels, a_pixels)]
        avg_diff = statistics.mean(diffs)

        b_clutter = _analyze_image_pixels(before)["clutter_score"]
        a_clutter = _analyze_image_pixels(after)["clutter_score"]
        clutter_reduction = max(0, b_clutter - a_clutter)

        score = min(1.0, (avg_diff * 0.6) + (clutter_reduction * 0.8))
        verified = score >= 0.35 or (clutter_reduction >= 0.15 and avg_diff >= 0.1)

        return {
            "verified": verified,
            "score": round(score, 2),
            "clutter_reduction": round(clutter_reduction, 2),
            "reason": "Cleanup verified" if verified else "Insufficient change detected",
        }
    except Exception as exc:
        return {"verified": False, "score": 0.0, "reason": str(exc)}


def predict_hotspots(historical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simple grid-based hotspot prediction from historical complaints."""
    if not historical:
        return []

    cell_size = 0.005
    grid: dict[tuple[int, int], dict[str, Any]] = {}

    for item in historical:
        lat, lon = item["latitude"], item["longitude"]
        key = (int(lat / cell_size), int(lon / cell_size))
        if key not in grid:
            grid[key] = {"lat_sum": 0.0, "lon_sum": 0.0, "count": 0}
        grid[key]["lat_sum"] += lat
        grid[key]["lon_sum"] += lon
        grid[key]["count"] += 1

    predictions = []
    for data in grid.values():
        count = data["count"]
        predictions.append(
            {
                "latitude": data["lat_sum"] / count,
                "longitude": data["lon_sum"] / count,
                "predicted_complaints_next_week": max(1, int(count * 1.2)),
                "risk_level": "High" if count >= 3 else "Medium" if count >= 2 else "Low",
            }
        )

    predictions.sort(key=lambda p: p["predicted_complaints_next_week"], reverse=True)
    return predictions[:8]
