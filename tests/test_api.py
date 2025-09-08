import subprocess
import time
import requests

CONTAINER_NAME = "test_api"
IMAGE_NAME = "dicksonml/realestate-api:latest"
BASE_URL = "http://127.0.0.1:8000"

def wait_for_api(url, timeout=120, interval=5):  # wait up to 2 minutes
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    return False

def setup_module(module):
    """Start the container before tests."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

    subprocess.run([
        "docker", "run", "-d", "-p", "8000:5000",
        "--name", CONTAINER_NAME, IMAGE_NAME
    ], check=True)

    if not wait_for_api(BASE_URL):
        subprocess.run(["docker", "logs", CONTAINER_NAME])
        assert False, "API did not become ready in time"

def teardown_module(module):
    """Stop container after tests."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

def test_root_endpoint():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    assert "Welcome" in r.text

def test_predict_endpoint():
    payload = {
        "transaction_date": 2013.167,   # example value
        "house_age": 10.0,
        "distance_to_mrt": 500.0,
        "convenience_stores": 5,
        "latitude": 24.982,
        "longitude": 121.543
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
