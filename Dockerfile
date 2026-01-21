FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for RDKit, visualization libraries, and curl for healthcheck
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libfreetype6-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for HF Spaces (required for security)
RUN useradd -m -u 1000 user

# Copy application code
COPY --chown=user:user . .

# Create the necessary directories with proper permissions
RUN mkdir -p /app/data /app/data/results /app/data/logs && \
    chown -R user:user /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HOME=/home/user
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV FRONTEND_PORT=7860

# Switch to non-root user
USER user

# Expose Streamlit port (7860 for HF Spaces) - backend runs internally on 8000
EXPOSE 7860

# Healthcheck - check both frontend and backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/_stcore/health && curl -f http://localhost:8000/api/v1/health || exit 1

# Make start script executable and run it
RUN chmod +x /app/start.sh

# Command to run both backend and frontend
CMD ["/bin/bash", "/app/start.sh"]
