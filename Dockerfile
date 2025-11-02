# Use a slim Python base
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src/ ./src
COPY models/ ./models

# Expose port
EXPOSE 8200

# Run the app with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8200"]
