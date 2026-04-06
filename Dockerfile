# Use official Python lightweight image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create and switch to non-root user
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Set working directory
WORKDIR /app

# Install dependencies before copying application code
# This optimizes Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=appuser:appgroup app/ app/

# Switch to the non-root user
USER appuser

# Expose port (Documentation only; Railway binds dynamically)
EXPOSE 8000

# Start server using dynamic $PORT binding
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
