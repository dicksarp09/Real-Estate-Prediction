import requests
import time
import subprocess

CONTAINER_NAME = "ci_test_api"
IMAGE_NAME = "dicksonml/realestate-api:latest"
BASE_URL = "http://127.0.0.1:8000"

def wait_for_api(url, timeout=30):
    """Wait until the API is up and responding."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

def setup_module(module):
    """Start the container before tests."""
    subprocess.run([
        "docker", "run", "-d", "-p", "8000:5000",
        "--name", CONTAINER_NAME, IMAGE_NAME
    ], check=True)

    assert wait_for_api(BASE_URL)

def teardown_module(module):
    """Stop and remove the container after tests."""
    subprocess.run(["docker", "stop", CONTAINER_NAME], check=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], check=True)

def test_root_endpoint():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    assert "message" in r.json()

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
