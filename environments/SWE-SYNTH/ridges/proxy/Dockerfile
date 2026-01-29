FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py .
COPY api.py .

# Expose port
EXPOSE 8000

# Set environment variables (can be overridden at runtime)
ENV OPENAI_BASE_URL=https://api.openai.com/v1

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

