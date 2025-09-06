import requests
import time
import subprocess

BASE_URL = "http://127.0.0.1:8000"
CONTAINER_NAME = "test_api"
IMAGE_NAME = "dicksonml/realestate-api:latest"

def wait_for_api(url, timeout=60, interval=3):
    """Wait for the API to be up and running."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    return False

def setup_module(module):
    """Start the container before tests."""
    # Remove any existing container with the same name
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)
    
    # Start container
    subprocess.run([
        "docker", "run", "-d", "-p", "8000:5000",
        "--name", CONTAINER_NAME, IMAGE_NAME
    ], check=True)
    
    assert wait_for_api(BASE_URL), "API did not become ready in time"

def teardown_module(module):
    """Stop and remove the container after tests."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

def test_root_endpoint():
    r = requests.get(BASE_URL)
    assert r.status_code == 200
    assert "Welcome" in r.json()["message"]

def test_predict_endpoint():
    payload = {
        "transaction_date": 2013.250,
        "house_age": 13.3,
        "distance_to_mrt": 561.9845,
        "convenience_stores": 5,
        "latitude": 24.98298,
        "longitude": 121.54024
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
