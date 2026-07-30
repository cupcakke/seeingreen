FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY config /app/config
COPY migrations /app/migrations
COPY alembic.ini /app/alembic.ini
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps .
RUN mkdir -p /app/var/artifacts /app/data
EXPOSE 8000
CMD ["uvicorn", "epistemic_uq.service:app", "--host", "0.0.0.0", "--port", "8000"]
