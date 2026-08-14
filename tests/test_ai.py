from clean_city_ai.ai import detect_waste, estimate_severity


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
