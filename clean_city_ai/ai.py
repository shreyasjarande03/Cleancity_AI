from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CleanCity.AI")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

YOLO_CONFIDENCE = 0.25
YOLO_IMGSZ = 640

# Custom YOLOv8 model classes trained on TACO dataset (10 classes)
YOLO_CLASSES = [
    "plastic_bottle",
    "plastic_wrapper",
    "plastic_container",
    "plastic_bag",
    "plastic_cap_lid",
    "metal_can",
    "metal_scrap",
    "glass",
    "paper_cardboard",
    "cigarette",
]

# Application-level mapping from TACO classes to CleanCity waste categories
TACO_TO_CLEANCITY_MAP = {
    "plastic_bottle": "Plastic",
    "plastic_wrapper": "Plastic",
    "plastic_container": "Plastic",
    "plastic_bag": "Plastic",
    "plastic_cap_lid": "Plastic",
    "metal_can": "Dry",
    "metal_scrap": "Dry",
    "glass": "Dry",
    "paper_cardboard": "Dry",
    "cigarette": "Mixed",
}

WASTE_KEYWORDS = {
    "Plastic": ["plastic", "bottle", "packaging", "wrapper", "bag", "container", "cup"],
    "Organic": ["food", "fruit", "vegetable", "organic", "garden", "leaf", "compost", "banana"],
    "Mixed": ["mixed", "trash", "garbage", "waste", "pile", "dump", "litter", "cigarette"],
    "Construction": ["brick", "cement", "construction", "debris", "concrete", "tile", "rubble"],
    "Dry": ["paper", "cardboard", "dry", "metal", "glass", "carton", "newspaper", "can"],
}

_yolo_model = None
_yolo_load_attempted = False


def _load_yolo() -> Any:
    """Load the custom TACO-trained YOLOv8 model once (singleton pattern)."""
    global _yolo_model, _yolo_load_attempted
    if _yolo_load_attempted:
        return _yolo_model
    _yolo_load_attempted = True

    logger.info("Checking custom YOLO model at path: %s (exists=%s)", MODEL_PATH, MODEL_PATH.exists())
    if not MODEL_PATH.exists():
        logger.warning(
            "Custom TACO YOLO model not found at '%s'. Falling back to pixel/keyword analysis.",
            MODEL_PATH,
        )
        _yolo_model = None
        return None

    try:
        from ultralytics import YOLO

        _yolo_model = YOLO(str(MODEL_PATH))
        logger.info("Successfully loaded custom YOLOv8 model from %s with classes: %s", MODEL_PATH, _yolo_model.names)
    except Exception as exc:
        logger.error("Failed to load YOLO model from %s: %s", MODEL_PATH, exc, exc_info=True)
        _yolo_model = None
    return _yolo_model


def get_dominant_waste_type(detections: list[dict[str, Any]]) -> str:
    """Determine the dominant CleanCity waste type using confidence-weighted aggregation."""
    if not detections:
        return "Mixed"

    weights: dict[str, float] = {}
    for item in detections:
        waste_type = item.get("waste_type", "Mixed")
        confidence = float(item.get("confidence", 0.0))
        weights[waste_type] = weights.get(waste_type, 0.0) + confidence

    if not weights:
        return "Mixed"
    dominant = max(weights, key=weights.get)
    logger.info("Dominant waste type aggregation scores: %s -> Dominant: %s", weights, dominant)
    return dominant


def generate_auto_description(
    waste_type: str,
    severity_level: str,
    detected_objects: list[dict[str, Any]] | None = None,
    clutter_score: float = 0.5,
) -> str:
    """Generate a clean, structured description summarizing the detected waste."""
    if detected_objects and len(detected_objects) > 0:
        counts: dict[str, int] = {}
        for obj in detected_objects:
            cls_name = obj.get("class", "waste item").replace("_", " ")
            counts[cls_name] = counts.get(cls_name, 0) + 1
        
        items_summary = ", ".join(f"{cnt} {name}{'s' if cnt > 1 and not name.endswith('s') else ''}" for name, cnt in counts.items())
        return f"{severity_level} severity {waste_type.lower()} waste detected: {items_summary}."
    
    if clutter_score > 0.35:
        return f"{severity_level} severity accumulation of {waste_type.lower()} waste requiring collection."
    return f"{severity_level} severity {waste_type.lower()} waste reported at this location."


def _analyze_image_pixels(image_path: str | Path) -> dict[str, Any]:
    """Fallback pixel-level image analysis when ML model is unavailable or detections are inconclusive."""
    try:
        from PIL import Image
        import statistics

        img = Image.open(image_path).convert("RGB")
        pixels = list(img.getdata())
        w, h = img.size
        total = max(1, w * h)

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
    except Exception as exc:
        logger.warning("Pixel analysis failed on image %s: %s", image_path, exc)
        return {"clutter_score": 0.5, "waste_type_hint": "Mixed", "brightness": 0.5}


def detect_waste_from_image(
    image_path: str | Path | None,
    image_description: str = "",
    user_category: str | None = None,
) -> dict[str, Any]:
    """Detect garbage using custom TACO YOLOv8 when available, with pixel and keyword fallbacks."""
    text = (image_description or "").lower()
    logger.info("=== START AI WASTE DETECTION ===")
    logger.info("Input parameters: image_path=%s, description='%s', user_category='%s'", image_path, text, user_category)

    if user_category and user_category.strip().title() in WASTE_KEYWORDS and not image_path:
        w_type = user_category.strip().title()
        sev = estimate_severity(text or w_type)
        desc = generate_auto_description(w_type, sev["level"])
        res = {
            "is_garbage": True,
            "waste_type": w_type,
            "confidence": 0.85,
            "method": "user",
            "suggested_description": desc,
            "severity": sev,
        }
        logger.info("User category provided without image: %s", res)
        return res

    if not image_path:
        res = detect_waste(text, user_category)
        sev = estimate_severity(text or res["waste_type"])
        res["severity"] = sev
        res["suggested_description"] = generate_auto_description(res["waste_type"], sev["level"])
        logger.info("No image provided, text-based detection result: %s", res)
        return res

    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file does not exist at %s. Falling back to text detection.", path)
        res = detect_waste(text, user_category)
        sev = estimate_severity(text or res["waste_type"])
        res["severity"] = sev
        res["suggested_description"] = generate_auto_description(res["waste_type"], sev["level"])
        return res

    model = _load_yolo()
    pixel_info = _analyze_image_pixels(path)
    yolo_inference_error = None

    if model is not None:
        try:
            logger.info("Running custom YOLOv8 inference on %s (imgsz=%d, conf=%.2f)", path, YOLO_IMGSZ, YOLO_CONFIDENCE)
            results = model(str(path), imgsz=YOLO_IMGSZ, conf=YOLO_CONFIDENCE, verbose=False)
            detections = []
            total_area = 0.0

            for result in results:
                names = result.names or {}
                for box in result.boxes:
                    cls_id = int(box.cls[0] if hasattr(box.cls, "__getitem__") else box.cls)
                    conf = float(box.conf[0] if hasattr(box.conf, "__getitem__") else box.conf)
                    raw_name = names.get(cls_id, YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else "unknown")
                    class_name = str(raw_name).lower()
                    clean_waste_type = TACO_TO_CLEANCITY_MAP.get(class_name, "Mixed")
                    
                    raw_xyxy = box.xyxy[0] if hasattr(box.xyxy, "__getitem__") else box.xyxy
                    if hasattr(raw_xyxy, "tolist"):
                        coords = raw_xyxy.tolist()
                    else:
                        coords = list(raw_xyxy)
                    xyxy = [round(float(c), 2) for c in coords]
                    
                    area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    total_area += area
                    detections.append(
                        {
                            "class": class_name,
                            "class_id": cls_id,
                            "confidence": round(conf, 2),
                            "waste_type": clean_waste_type,
                            "bbox": xyxy,
                            "area": area,
                        }
                    )

            logger.info("YOLO inference completed. Objects detected count: %d", len(detections))
            for i, d in enumerate(detections):
                logger.info("  Detection #%d: class=%s (ID=%d), conf=%.2f, mapped_type=%s, bbox=%s", i + 1, d["class"], d["class_id"], d["confidence"], d["waste_type"], d["bbox"])

            if detections:
                dominant_waste_type = get_dominant_waste_type(detections)
                best_detection = max(detections, key=lambda d: d["confidence"])
                confidence = min(0.99, best_detection["confidence"] + 0.05)

                img_w, img_h = _get_image_size(path)
                area_ratio = total_area / max(1, img_w * img_h)
                severity = estimate_severity_from_signals(
                    area_ratio=area_ratio,
                    clutter_score=pixel_info["clutter_score"],
                    description=text,
                    num_detections=len(detections),
                )

                final_waste_type = user_category.strip().title() if user_category and user_category.strip() else dominant_waste_type
                formatted_objects = [
                    {
                        "class": d["class"],
                        "class_id": d["class_id"],
                        "confidence": d["confidence"],
                        "waste_type": d["waste_type"],
                        "bbox": d["bbox"],
                    }
                    for d in detections
                ]
                auto_desc = generate_auto_description(final_waste_type, severity["level"], formatted_objects, pixel_info["clutter_score"])

                res = {
                    "is_garbage": True,
                    "waste_type": final_waste_type,
                    "confidence": round(confidence, 2),
                    "method": "YOLOv8",
                    "detections": len(detections),
                    "detected_objects": formatted_objects,
                    "severity": severity,
                    "suggested_description": auto_desc,
                }
                logger.info("Final YOLO detection result: %s", res)
                return res
            else:
                logger.info("No YOLO detections above confidence threshold %.2f.", YOLO_CONFIDENCE)
        except Exception as exc:
            yolo_inference_error = str(exc)
            logger.error("YOLO inference failed: %s. Check server logs.", exc, exc_info=True)

    # Fallback path if YOLO had an error or found no objects
    clutter = pixel_info["clutter_score"]
    logger.info("Evaluating fallback logic: clutter_score=%.2f, text='%s'", clutter, text)

    if clutter < 0.15 and ("clean" in text or "no garbage" in text):
        res = {"is_garbage": False, "waste_type": "Unknown", "confidence": 0.88, "method": "pixels", "suggested_description": "Clean area with no garbage detected."}
        logger.info("Scene classified as clean: %s", res)
        return res

    keyword_result = detect_waste(text or "garbage waste", user_category)
    waste_type = keyword_result["waste_type"]
    if waste_type == "Unknown" or waste_type == "Mixed":
        waste_type = pixel_info["waste_type_hint"]

    severity = estimate_severity_from_signals(clutter * 0.4, clutter, text)
    final_waste_type = user_category.strip().title() if user_category and user_category.strip() else waste_type
    auto_desc = generate_auto_description(final_waste_type, severity["level"], None, clutter)
    res = {
        "is_garbage": True,
        "waste_type": final_waste_type,
        "confidence": round(0.6 + clutter * 0.3, 2),
        "method": "pixels+keywords",
        "detections": 0,
        "detected_objects": [],
        "severity": severity,
        "suggested_description": auto_desc,
    }
    if yolo_inference_error:
        res["error"] = "AI detection failed. Please check the server logs."
        logger.warning("AI detection returned with error notification: %s", yolo_inference_error)
    logger.info("Fallback detection result: %s", res)
    return res


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


def estimate_severity_from_signals(
    area_ratio: float,
    clutter_score: float,
    description: str,
    num_detections: int = 0,
) -> dict[str, Any]:
    text = (description or "").lower()
    detection_factor = min(0.3, num_detections * 0.05)
    score = max(area_ratio + detection_factor, clutter_score * 0.5)

    if any(kw in text for kw in ["critical", "drainage", "block", "overflow"]):
        level = "Critical"
        score = max(score, 0.95)
    elif score > 0.35 or any(kw in text for kw in ["large", "pile", "dump", "big", "heap"]) or num_detections >= 5:
        level = "High"
        score = max(score, 0.85)
    elif score > 0.15 or any(kw in text for kw in ["medium", "several", "roadside"]) or num_detections >= 2:
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
