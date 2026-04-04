FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "schema_migration_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
