# CleanCity AI

CleanCity AI is a smart garbage reporting and collection management system designed for urban waste monitoring.

## Features

- Citizen complaint reporting with photo upload and automatic location capture
- AI-assisted garbage detection and waste classification
- Severity scoring for prioritization
- Smart collector assignment based on proximity
- Collector task dashboard and status workflow
- Admin analytics dashboard
- SQLite-based local data storage for a complete working prototype

## Project structure

- `app/` — web application and dashboard
- `clean_city_ai/` — business logic, AI, and routing
- `data/` — local SQLite database and seed data
- `tests/` — unit tests for AI logic

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m clean_city_ai.app
```

Then open `http://localhost:8000` in a browser.

## Demo accounts

- Admin: `admin@cleancity.ai` / `admin123`
- Collector: `collector@cleancity.ai` / `collector123`
- Citizen: `citizen@cleancity.ai` / `citizen123`
