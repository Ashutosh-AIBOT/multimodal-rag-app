# Dockerfile - Production Ready for Hugging Face
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && echo "✅ System dependencies installed"

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    && echo "✅ Python dependencies installed"

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/chroma_db \
             /app/data/uploads \
             /app/data/extracted/images \
             /app/data/extracted/tables \
    && echo "✅ Directories created"

# Create non-root user and set proper ownership
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app \
    && echo "✅ User created"

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 7860

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_ENABLE_CORS=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]