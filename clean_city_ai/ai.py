from __future__ import annotations

from typing import Dict, Any


WASTE_KEYWORDS = {
    "Plastic": ["plastic", "bottle", "packaging", "wrapper", "bag", "container"],
    "Organic": ["food", "fruit", "vegetable", "organic", "garden", "leaf", "compost"],
    "Mixed": ["mixed", "trash", "garbage", "waste", "pile", "dump"],
    "Construction": ["brick", "cement", "construction", "debris", "concrete", "tile"],
    "Dry": ["paper", "cardboard", "dry", "metal", "glass", "carton"],
}


def detect_waste(image_description: str, user_category: str | None = None) -> Dict[str, Any]:
    """Rule-based waste detection for a demo AI module.

    In a production deployment, this would call a trained computer-vision model.
    """
    text = (image_description or "").lower()
    if not text or "clean" in text or "no garbage" in text or "none" in text:
        return {"is_garbage": False, "waste_type": "Unknown", "confidence": 0.92}

    selected_type = user_category.strip().title() if user_category and user_category.strip() else None
    if selected_type in WASTE_KEYWORDS:
        return {"is_garbage": True, "waste_type": selected_type, "confidence": 0.96}

    for waste_type, keywords in WASTE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return {"is_garbage": True, "waste_type": waste_type, "confidence": 0.9}

    return {"is_garbage": True, "waste_type": "Mixed", "confidence": 0.72}


def estimate_severity(description: str) -> Dict[str, Any]:
    """Assign a severity score based on the complaint description."""
    text = (description or "").lower()

    if any(keyword in text for keyword in ["large", "pile", "dump", "block", "overflow", "big"]):
        level = "High"
        score = 0.88
    elif any(keyword in text for keyword in ["medium", "several", "collection", "roadside"]):
        level = "Medium"
        score = 0.62
    else:
        level = "Low"
        score = 0.35

    if "critical" in text or "drainage" in text:
        level = "Critical"
        score = 0.95

    return {"level": level, "score": score}
