import pytest
import sys
import os

# Récupère le chemin du répertoire courant
current_directory = os.path.abspath(os.path.dirname(__file__))

# Vérifie si le répertoire courant n'est pas déjà dans sys.path
if current_directory not in sys.path:
    # Ajoute le répertoire courant à la liste des chemins de recherche
    sys.path.insert(0, current_directory)

from main import get_prediction
from unittest.mock import patch, MagicMock
from schemas import PredictionData
from fastapi import HTTPException

@pytest.fixture
def prediction_data():
    return PredictionData(
        dataset_path="test_dataset.xlsx",
        images_path="test_images/",
        tokenizer_config_path="test_tokenizer/",
        lstm_model_path="test_lstm/",
        vgg16_model_path="test_vgg16/",
        model_weights_path="test_weights/",
        mapper_path="test_mapper/"
    )

@pytest.mark.asyncio
@patch("keras.preprocessing.text.tokenizer_from_json")
@patch("keras.models.load_model")
@patch("json.load")
@patch("builtins.open")
async def test_get_prediction_success(mock_open, mock_json_load, mock_load_model, mock_tokenizer_from_json, prediction_data):
    # Mocking data
    mock_open.return_value.__enter__.return_value.read.return_value = {"config": {"word_counts": {"word": 1}}}
    mock_load_model.return_value = MagicMock()
    mock_json_load.return_value = {"weights": [0.5, 0.5]}
    mock_tokenizer_from_json.return_value = {"config": {"word_counts": {"word": 1}}}
   
    # Call the function
    predictions = await get_prediction(prediction_data)

    # Assertions
    assert isinstance(predictions, dict)

@pytest.mark.asyncio
@patch("keras.preprocessing.text.tokenizer_from_json")
@patch("keras.models.load_model")
@patch("json.load")
@patch("builtins.open")
async def test_get_prediction_ioerror(mock_open, mock_json_load, mock_load_model, mock_tokenizer_from_json, prediction_data):
    # Mocking IOError
    mock_open.side_effect = IOError

    # Call the function and check for HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await get_prediction(prediction_data)
    
    assert exc_info.value.status_code == 404
    assert "File not found" in exc_info.value.detail

@pytest.mark.asyncio
@patch("keras.preprocessing.text.tokenizer_from_json")
@patch("keras.models.load_model")
@patch("json.load")
@patch("builtins.open")
async def test_get_prediction_generic_error(mock_open, mock_json_load, mock_load_model, mock_tokenizer_from_json, prediction_data):
    # Mocking generic error
    mock_open.return_value.__enter__.return_value.read.return_value = '{"token": "some_token"}'
    mock_load_model.side_effect = Exception

    # Call the function and check for HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await get_prediction(prediction_data)
    
    assert exc_info.value.status_code == 404
    assert "Error" in exc_info.value.detail

