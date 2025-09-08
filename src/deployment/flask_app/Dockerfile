FROM python:3.9-slim

WORKDIR /app

# Copy root requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY src /app/src
COPY models /app/models
COPY src/deployment/flask_app /app

# Expose Flask port
EXPOSE 5000

# Set Python path
ENV PYTHONPATH=/app

# Run Flask app
CMD ["python", "app.py"]

CMD ["python", "app.py"]
