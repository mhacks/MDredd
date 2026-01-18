# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Sync dependencies
RUN uv sync --frozen --no-cache

# Expose port
EXPOSE 8000

# Run app
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
