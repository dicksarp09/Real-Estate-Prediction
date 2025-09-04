FROM python:3.9-slim

WORKDIR /app

COPY src/deployment/flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src folder
COPY src /app/src

# Copy flask app
COPY src/deployment/flask_app /app

# Copy model files (important!)
COPY models /app/models

# Expose Flask port
EXPOSE 5000

# Set Python path
ENV PYTHONPATH=/app

# Run Flask app
CMD ["python", "app.py"]
