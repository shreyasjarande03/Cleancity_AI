# CleanCity AI

**CleanCity AI: An Intelligent Citizen-Driven Waste Detection and Smart Collection Management System**

CleanCity AI is an AI-powered smart waste management platform that enables citizens to report garbage through photographs. The system uses computer vision to detect and classify waste, GPS to identify its location, and intelligent task allocation to assign complaints to nearby garbage collectors.

## Features

### Citizen Portal
- Register/login with role-based access
- Report garbage with photo upload
- Automatic GPS location capture (browser Geolocation API)
- Interactive map to adjust report location
- AI waste detection and classification
- Track complaint status in real time
- Duplicate reports at the same location are merged automatically

### AI Pipeline
- **YOLOv8** object detection (with pixel/keyword fallback if model unavailable)
- Waste classification: Plastic, Organic, Mixed, Construction, Dry
- Severity scoring: Low → Medium → High → Critical
- Priority score based on severity + duplicate report count
- Before/after image verification for cleanup confirmation

### Collector Dashboard
- View assigned complaints sorted by priority
- See garbage locations on OpenStreetMap
- Workflow: Accept → Start Cleaning → Upload After Photo → Complete
- AI verifies before/after photos automatically

### Admin Dashboard
- Analytics: total, pending, assigned, in progress, completed, high priority
- Complaint map with severity markers
- Garbage hotspot clustering from historical data
- Hotspot prediction for preventive cleaning

### REST API
- `GET /api/health` — service health
- `GET /api/complaints` — all complaints
- `GET /api/complaints/{id}` — complaint detail
- `GET /api/hotspots` — hotspot clusters and predictions
- `GET /api/stats` — admin statistics

## Project Structure

```
CleanCity_AI/
├── clean_city_ai/
│   ├── app.py          # FastAPI routes and web UI
│   ├── ai.py           # YOLO detection, severity, verification, hotspots
│   └── database.py     # SQLite data layer
├── templates/          # Citizen, Collector, Admin dashboards
├── static/
│   ├── styles.css
│   └── uploads/        # Uploaded complaint photos
├── tests/              # Unit tests
├── data/               # SQLite database (auto-created)
└── requirements.txt
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m clean_city_ai.app
```

Open **http://localhost:8000** in your browser.

> On first run, YOLOv8 will download `yolov8n.pt` (~6 MB). If Ultralytics is unavailable, the system falls back to pixel analysis + keyword classification.

## Demo Accounts

| Role      | Email                    | Password      |
|-----------|--------------------------|---------------|
| Citizen   | citizen@cleancity.ai     | citizen123    |
| Collector | collector@cleancity.ai   | collector123  |
| Admin     | admin@cleancity.ai       | admin123      |

## Complaint Workflow

```
Pending → Assigned → In Progress → Cleaned → Verified
```

1. Citizen submits photo + GPS
2. AI detects waste type and severity
3. Nearest available collector is assigned
4. Collector accepts and starts cleaning
5. Collector uploads after-cleaning photo
6. AI compares before/after images → Verified or Cleaned

## Run Tests

```bash
pytest tests/ -v
```

## Technology Stack

| Component    | Technology                          |
|-------------|--------------------------------------|
| Backend     | Python + FastAPI                     |
| AI/ML       | YOLOv8 (Ultralytics) + Pillow        |
| Database    | SQLite (local prototype)             |
| Maps        | Leaflet + OpenStreetMap              |
| Frontend    | Jinja2 HTML templates                |
| Image Store | Local filesystem (`static/uploads/`) |

## Advanced Features Implemented

- ✅ AI garbage detection (YOLO + fallback)
- ✅ Duplicate complaint detection (~50 m radius)
- ✅ Smart collector allocation (Haversine distance)
- ✅ Priority prediction (severity + report count)
- ✅ Before/after verification (image comparison)
- ✅ Garbage hotspot prediction (grid clustering)
