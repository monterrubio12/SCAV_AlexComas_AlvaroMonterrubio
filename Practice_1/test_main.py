from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_run_length_encode():
    response = client.post("/run_length_encode/", json={"data": [1, 1, 3, 3, 4, 4, 5, 6]})
    assert response.status_code == 200
    assert response.json()["encoded"] == [[1, 2], [3, 2], [4, 2], [5, 1], [6, 1]]

def test_dwt_transform():
    input_data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    response = client.post("/dwt_transform/", json={"data": input_data})
    assert response.status_code == 200
    assert "transformed" in response.json()
