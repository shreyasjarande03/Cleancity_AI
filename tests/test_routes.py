from fastapi.testclient import TestClient
from clean_city_ai.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_login_and_admin_portal():
    login_resp = client.post(
        "/login",
        data={"email": "admin@cleancity.ai", "password": "admin123"},
        follow_redirects=False,
    )
    assert login_resp.status_code == 303
    assert login_resp.headers["location"] == "/admin"

    # Access admin dashboard with cookie
    admin_resp = client.get("/admin", cookies={"user_email": "admin@cleancity.ai"})
    assert admin_resp.status_code == 200
    assert "Admin Dashboard" in admin_resp.text


def test_collector_portal():
    collector_resp = client.get("/collector", cookies={"user_email": "collector@cleancity.ai"})
    assert collector_resp.status_code == 200
    assert "Collector Dashboard" in collector_resp.text


def test_citizen_portal():
    citizen_resp = client.get("/citizen", cookies={"user_email": "citizen@cleancity.ai"})
    assert citizen_resp.status_code == 200
    assert "Citizen Portal" in citizen_resp.text


def test_api_endpoints():
    assert client.get("/api/complaints").status_code == 200
    assert client.get("/api/hotspots").status_code == 200
    assert client.get("/api/stats").status_code == 200
