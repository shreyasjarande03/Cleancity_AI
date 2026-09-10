from pathlib import Path
from unittest.mock import MagicMock, patch

from clean_city_ai.ai import (
    MODEL_PATH,
    TACO_TO_CLEANCITY_MAP,
    YOLO_CLASSES,
    YOLO_CONFIDENCE,
    _load_yolo,
    compute_priority_score,
    detect_waste,
    detect_waste_from_image,
    estimate_severity,
    estimate_severity_from_signals,
    get_dominant_waste_type,
    predict_hotspots,
    verify_cleanup,
)


def test_model_path_configuration():
    assert MODEL_PATH.name == "best.pt"
    assert MODEL_PATH.parent.name == "models"


def test_taco_classes_specification():
    expected_classes = [
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
    assert len(YOLO_CLASSES) == 10
    assert YOLO_CLASSES == expected_classes


def test_taco_to_cleancity_mapping():
    # Plastic category
    assert TACO_TO_CLEANCITY_MAP["plastic_bottle"] == "Plastic"
    assert TACO_TO_CLEANCITY_MAP["plastic_wrapper"] == "Plastic"
    assert TACO_TO_CLEANCITY_MAP["plastic_container"] == "Plastic"
    assert TACO_TO_CLEANCITY_MAP["plastic_bag"] == "Plastic"
    assert TACO_TO_CLEANCITY_MAP["plastic_cap_lid"] == "Plastic"

    # Dry category
    assert TACO_TO_CLEANCITY_MAP["metal_can"] == "Dry"
    assert TACO_TO_CLEANCITY_MAP["metal_scrap"] == "Dry"
    assert TACO_TO_CLEANCITY_MAP["glass"] == "Dry"
    assert TACO_TO_CLEANCITY_MAP["paper_cardboard"] == "Dry"

    # Mixed category
    assert TACO_TO_CLEANCITY_MAP["cigarette"] == "Mixed"


def test_yolo_confidence_threshold():
    assert YOLO_CONFIDENCE == 0.25


def test_get_dominant_waste_type_single():
    detections = [{"waste_type": "Plastic", "confidence": 0.85}]
    assert get_dominant_waste_type(detections) == "Plastic"


def test_get_dominant_waste_type_weighted_aggregation():
    detections = [
        {"waste_type": "Plastic", "confidence": 0.82},
        {"waste_type": "Plastic", "confidence": 0.71},
        {"waste_type": "Dry", "confidence": 0.95},
    ]
    # Plastic sum = 1.53, Dry sum = 0.95 -> Dominant is Plastic
    assert get_dominant_waste_type(detections) == "Plastic"


def test_get_dominant_waste_type_empty():
    assert get_dominant_waste_type([]) == "Mixed"


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
    result = estimate_severity_from_signals(0.5, 0.6, "large pile near drainage", num_detections=4)
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


def test_yolo_detection_mocked_flow(tmp_path):
    # Create a dummy image file
    from PIL import Image

    dummy_image = tmp_path / "test_garbage.jpg"
    img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    img.save(dummy_image)

    # Mock YOLO model box output
    mock_box1 = MagicMock()
    mock_box1.cls = [0]  # plastic_bottle
    mock_box1.conf = [0.85]
    mock_box1.xyxy = [[10.0, 10.0, 100.0, 100.0]]

    mock_box2 = MagicMock()
    mock_box2.cls = [5]  # metal_can
    mock_box2.conf = [0.70]
    mock_box2.xyxy = [[200.0, 200.0, 250.0, 250.0]]

    mock_result = MagicMock()
    mock_result.names = {0: "plastic_bottle", 5: "metal_can"}
    mock_result.boxes = [mock_box1, mock_box2]

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]

    with patch("clean_city_ai.ai._load_yolo", return_value=mock_model):
        res = detect_waste_from_image(dummy_image, "bottles and cans on road")
        assert res["is_garbage"] is True
        assert res["method"] in {"yolo", "YOLOv8"}
        assert res["detections"] == 2
        assert len(res["detected_objects"]) == 2
        assert res["detected_objects"][0]["class"] == "plastic_bottle"
        assert res["detected_objects"][0]["waste_type"] == "Plastic"
        assert res["detected_objects"][1]["class"] == "metal_can"
        assert res["detected_objects"][1]["waste_type"] == "Dry"
        assert res["waste_type"] == "Plastic"  # Dominant type


def test_yolo_fallback_when_model_fails_or_none(tmp_path):
    from PIL import Image

    dummy_image = tmp_path / "test_fallback.jpg"
    img = Image.new("RGB", (100, 100), color=(80, 80, 80))
    img.save(dummy_image)

    with patch("clean_city_ai.ai._load_yolo", return_value=None):
        res = detect_waste_from_image(dummy_image, "plastic wrappers scattered")
        assert res["is_garbage"] is True
        assert res["waste_type"] == "Plastic"
        assert "pixels" in res["method"] or "keywords" in res["method"]


def test_detect_waste_from_base64_data_url():
    import base64
    import io
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(60, 60, 60))
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    with patch("clean_city_ai.ai._load_yolo", return_value=None):
        res = detect_waste_from_image(data_url, "organic vegetable scraps", "Organic")
        assert res["is_garbage"] is True
        assert res["waste_type"] == "Organic"


def test_verify_cleanup_with_data_urls():
    import base64
    import io
    from PIL import Image

    # Dirty image
    buf_b = io.BytesIO()
    img_b = Image.new("RGB", (64, 64), color=(50, 50, 50))
    img_b.save(buf_b, format="JPEG")
    url_b = f"data:image/jpeg;base64,{base64.b64encode(buf_b.getvalue()).decode('utf-8')}"

    # Clean image
    buf_a = io.BytesIO()
    img_a = Image.new("RGB", (64, 64), color=(220, 220, 220))
    img_a.save(buf_a, format="JPEG")
    url_a = f"data:image/jpeg;base64,{base64.b64encode(buf_a.getvalue()).decode('utf-8')}"

    verification = verify_cleanup(url_b, url_a)
    assert "verified" in verification
    assert "score" in verification
    assert verification["verified"] is True

