# syntax =docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 300 --retries 10 -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]