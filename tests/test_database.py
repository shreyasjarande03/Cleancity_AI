from clean_city_ai.database import (
    find_duplicate_complaint,
    get_admin_stats,
    get_all_complaints,
    get_complaints_for_citizen,
    get_complaints_for_collector,
    haversine_km,
    increment_duplicate_report,
    init_db,
    save_complaint,
    update_complaint_status,
)


def test_haversine_km():
    dist = haversine_km(12.9716, 77.5946, 12.9716, 77.5946)
    assert dist == 0.0


def test_collector_filtering_excludes_cleaned():
    init_db()
    complaints = get_complaints_for_collector(1)
    for c in complaints:
        assert c["status"] not in ("Verified", "Cleaned")
