# Production image with single stage
FROM python:3.12-slim

# Install runtime and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY --chown=appuser:appuser ../requirements.txt /app/requirements.txt

# Copy application files with proper ownership
COPY --chown=appuser:appuser app/ /app/

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PATH="/home/appuser/venv/bin:$PATH"

# Switch to non-root user
USER appuser

# Create virtual environment and install dependencies as appuser
RUN python -m venv /home/appuser/venv && \
    /home/appuser/venv/bin/pip install --no-cache-dir --upgrade pip wheel && \
    /home/appuser/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /home/appuser/venv/bin/pip install --no-cache-dir gunicorn

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]