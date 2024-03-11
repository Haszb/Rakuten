import pytest

import sys
import os


# Récupère le chemin du répertoire courant
current_directory = os.path.abspath(os.path.dirname(__file__))

# Vérifie si le répertoire courant n'est pas déjà dans sys.path
if current_directory not in sys.path:
    # Ajoute le répertoire courant à la liste des chemins de recherche
    sys.path.insert(0, current_directory)

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, status
from main import get_prediction
from schemas import PredictionData, User
import db_models


valid_user = User(id=1, is_active=True, username="DataScientest", email="admin@datascientest.com", role="Admin")
unauthorized_user = User(id=2, is_active=True, username="DataScientest 2", email="client@datascientest.com", role="Client")


@pytest.mark.asyncio
async def test_get_prediction_invalid_file_valid_user():
    prediction_data = PredictionData(
        dataset_path="invalid_dataset_path.xlsx",
        images_path="invalid_images_path/",
        tokenizer_config_path="invalid_tokenizer_path/",
        lstm_model_path="invalid_lstm_path/",
        vgg16_model_path="invalid_vgg16_path/",
        model_weights_path="invalid_weights_path/",
        mapper_path="invalid_mapper_path/"
    )
    #valid_user = User(role="Admin", id=1, is_active=True, username="DataScientest", email="admin@datascientest.com")
    with pytest.raises(HTTPException) as exc_info:
        await get_prediction(prediction_data, current_user=valid_user)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert valid_user.role in [db_models.Role.admin, db_models.Role.employe]


@pytest.mark.asyncio
async def test_get_prediction_io_error():
    prediction_data = PredictionData(
        dataset_path="test_dataset_path.xlsx",
        images_path="test_images_path/",
        tokenizer_config_path="test_tokenizer_path/",
        lstm_model_path="test_lstm_path/",
        vgg16_model_path="test_vgg16_path/",
        model_weights_path="test_weights_path/",
        mapper_path="test_mapper_path/"
    )
    with pytest.raises(HTTPException) as exc_info:
        await get_prediction(prediction_data, current_user=valid_user)
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_prediction_unauthorized_user():
    prediction_data = PredictionData(
        dataset_path="test_dataset_path.xlsx",
        images_path="test_images_path/",
        tokenizer_config_path="test_tokenizer_path/",
        lstm_model_path="test_lstm_path/",
        vgg16_model_path="test_vgg16_path/",
        model_weights_path="test_weights_path/",
        mapper_path="test_mapper_path/"
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_prediction(prediction_data, current_user=unauthorized_user)
    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_get_prediction_invalid_user_role():
    prediction_data = PredictionData(
        dataset_path="test_dataset_path.xlsx",
        images_path="test_images_path/",
        tokenizer_config_path="test_tokenizer_path/",
        lstm_model_path="test_lstm_path/",
        vgg16_model_path="test_vgg16_path/",
        model_weights_path="test_weights_path/",
        mapper_path="test_mapper_path/"
    )
    invalid_user = MagicMock()
    invalid_user.role = "invalid_role"
    with pytest.raises(HTTPException) as exc_info:
        await get_prediction(prediction_data, current_user=invalid_user)
    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN