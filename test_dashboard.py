import pytest
import requests
from dashboard import get_prediction

def test_get_prediction(monkeypatch):
    # Simulation d'une réponse réussie de l'API
    def mock_post_success(*args, **kwargs):
        # Classe MockResponse qui simule la réponse de l'API
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data
                self.text = str(json_data)
    
            def json(self):
                return self.json_data
    
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError(f"{self.status_code} Error")
    
        return MockResponse(200, {"prediction": 0.75})

    monkeypatch.setattr(requests, 'post', mock_post_success)

    # Test de récupération de prédiction réussie
    assert get_prediction(123) == 0.75

    # Simulation d'une réponse échouée de l'API
    def mock_post_failure(*args, **kwargs):
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data
                self.text = str(json_data)
    
            def json(self):
                return self.json_data
    
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError(f"{self.status_code} Error")
    
        return MockResponse(400, {"error": "Invalid request"})

    monkeypatch.setattr(requests, 'post', mock_post_failure)

    # Test de récupération de prédiction échouée
    assert get_prediction(123) is None

if __name__ == '__main__':
    pytest.main()

