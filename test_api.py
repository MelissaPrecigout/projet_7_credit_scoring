import pytest
from flask import Flask
import json
from api import app, test_data

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == "API pour prédire l'accord d'un prêt"

def test_check_client_id_exists(client):
    # Supposons que le client_id 1 existe dans les données test
    response = client.get('/check_client/1')
    assert response.status_code == 200
    assert response.json is True

def test_check_client_id_not_exists(client):
    # Supposons que le client_id 9999 n'existe pas dans les données test
    response = client.get('/check_client/9999')
    assert response.status_code == 200
    assert response.json is False

def test_get_client_info(client):
    # Supposons que le client_id 1 existe dans les données test
    existing_client_id = 1
    response = client.get(f'/client_info/{existing_client_id}')
    assert response.status_code == 200
    assert 'client_id' in response.json
    assert response.json['client_id'] == existing_client_id

def test_get_client_info_not_found(client):
    non_existing_client_id = 9999
    response = client.get(f'/client_info/{non_existing_client_id}')
    assert response.status_code == 404
    assert response.json == {"error": "Client not found"}

def test_update_client_info(client):
    existing_client_id = 1
    updated_data = {"DAYS_BIRTH": 85}
    response = client.put(f'/client_info/{existing_client_id}', json=updated_data)
    assert response.status_code == 200
    assert response.json == {"message": "Client information updated"}

def test_update_client_info_not_found(client):
    non_existing_client_id = 9999
    updated_data = {"DAYS_BIRTH": 85}
    response = client.put(f'/client_info/{non_existing_client_id}', json=updated_data)
    assert response.status_code == 404
    assert response.json == {"error": "Client not found"}

def test_submit_new_client(client):
    new_client_data = {"DAYS_BIRTH": 85, "EXT_SOURCE_1": 0.502129}
    response = client.post('/client_info', json=new_client_data)
    assert response.status_code == 201
    assert "message" in response.json
    assert response.json["message"] == "New client submitted"
    assert "client_id" in response.json

def test_get_prediction(client):
    # Supposons que le client_id 1 existe dans les données test
    response = client.post('/prediction', json={'client_id': 1})
    assert response.status_code == 200
    assert 'prediction' in response.json

def test_get_prediction_no_client_id(client):
    response = client.post('/prediction', json={})
    assert response.status_code == 400
    assert response.json == {"error": "client_id is required"}

def test_get_prediction_client_not_found(client):
    # Supposons que le client_id 9999 n'existe pas dans les données test
    response = client.post('/prediction', json={'client_id': 9999})
    assert response.status_code == 404
    assert response.json == {"error": "Client not found"}
