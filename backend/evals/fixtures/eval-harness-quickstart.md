# Quickstart Guide

## Environment

- Install Python 3.11 before running the backend locally.
- Create a virtual environment under `backend/.venv`.
- Install dependencies from `backend/requirements.txt`.

## First Run

1. Copy `backend/.env.example` to `backend/.env`.
2. Fill in the model API keys.
3. Start the service with `backend/.venv/Scripts/python.exe backend/run.py`.

## Verification

- Open `http://127.0.0.1:8000/docs` after startup.
- Use the `/health` endpoint to verify the service is ready.
