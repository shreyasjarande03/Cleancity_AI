from clean_city_ai.ai import (
    compute_priority_score,
    detect_waste,
    detect_waste_from_image,
    estimate_severity,
    estimate_severity_from_signals,
    predict_hotspots,
    verify_cleanup,
)


def test_detect_waste_recognizes_plastic():
    result = detect_waste("plastic bottle in a street", "plastic")
    assert result["is_garbage"] is True
    assert result["waste_type"] == "Plastic"


def test_detect_waste_detects_no_garbage_for_clean_scene():
    result = detect_waste("clean road with no litter", "none")
    assert result["is_garbage"] is False
    assert result["waste_type"] == "Unknown"


def test_severity_scales_with_amount():
    low = estimate_severity("small scattered waste")
    high = estimate_severity("large garbage pile covering road")
    assert high["level"] in {"High", "Critical"}
    assert low["level"] in {"Low", "Medium"}
    assert high["score"] >= low["score"]


def test_severity_from_signals():
    result = estimate_severity_from_signals(0.5, 0.6, "large pile near drainage")
    assert result["level"] in {"High", "Critical"}
    assert result["score"] >= 0.5


def test_priority_score_increases_with_reports():
    base = compute_priority_score(0.5, 1)
    boosted = compute_priority_score(0.5, 5)
    assert boosted > base


def test_hotspot_prediction():
    historical = [
        {"latitude": 12.9716, "longitude": 77.5946},
        {"latitude": 12.9717, "longitude": 77.5947},
        {"latitude": 12.9800, "longitude": 77.6000},
    ]
    predictions = predict_hotspots(historical)
    assert len(predictions) >= 1
    assert "risk_level" in predictions[0]


def test_detect_waste_from_image_without_file():
    result = detect_waste_from_image(None, "plastic bottles on roadside", "Plastic")
    assert result["is_garbage"] is True
    assert result["waste_type"] == "Plastic"


def test_verify_cleanup_missing_images():
    result = verify_cleanup(None, None)
    assert result["verified"] is False
