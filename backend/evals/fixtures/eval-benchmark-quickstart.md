# Benchmark Quickstart Guide

## Runtime Requirements

- Install Python 3.11 before running the backend locally.
- Create the virtual environment at `backend/.venv`.
- Install dependencies from `backend/requirements.txt`.

## Configuration

1. Copy `backend/.env.example` to `backend/.env`.
2. Fill in the model API keys before the first run.
3. Keep `AI_RAG_APP__ACTIVE_SCENE=generic_assistant` for document RAG checks.

## Startup And Verification

- Start the service with `backend/.venv/Scripts/python.exe backend/run.py`.
- After startup, open `http://127.0.0.1:8000/docs`.
- Use the `/health` endpoint to confirm the service is ready.

