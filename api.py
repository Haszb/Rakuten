from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel

from src.features.build_features import TextPreprocessor
from src.features.build_features import ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd
import argparse

class Predict:
    def __init__(
        self,
        tokenizer,
        lstm,
        vgg16,
        best_weights,
        mapper,
        filepath,
        imagepath
    ):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.filepath = filepath
        self.imagepath = imagepath

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self):
        X = pd.read_csv(self.filepath)
        
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.imagepath)
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

        sequences = self.tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        lstm_proba = self.lstm.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        concatenate_proba = (
            self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)

        return {
            i: self.mapper[str(final_predictions[i])]
            for i in range(len(final_predictions))
        }

api = FastAPI(
    title="Rakuten API",
    description="Our RAKUTEN API Project.",
    version="1.0")


class PredictionData(BaseModel):
    """
    A class to represent a Vehicle.
    
    Attributes:
    -----------
    dataset_path: str
        The path where the excel file is saved.
    images_path: str
        The folder path where to find the images (which should be labelled properly).
    tokenizer_config_path: str
        The folder path where the tockenize file is saved.
    lstm_model_path: str
        The folder path where the lstm model is saved.
    vgg16_model_path: str
        The folder path where the vgg16 model is saved.
    model_weights_path: str
        The folder path where the file having the models weights is saved.
    mapper_path: str
        The folder path where the mapper file is saved.
    """
    dataset_path: str
    images_path: str
    tokenizer_config_path:Optional[str] = "models/" 
    lstm_model_path:Optional[str] = "models/"
    vgg16_model_path:Optional[str] = "models/" 
    model_weights_path:Optional[str] = "models/"
    mapper_path:Optional[str] = "models/"

@api.get('/ping')
async def get_ping():
    return "It's working"

@api.post('/prediction')
async def get_prediction(prediction_data:PredictionData):
    """
    Return the prediction of the product category.

    Parameters:
    -----------
        prediction_data: PredictionData
            PredictionData object which has the the excel file path for products to categorize and also the images path.
    """
    try:
        # Charger les configurations et modèles
        with open(prediction_data.tokenizer_config_path + "tokenizer_config.json", "r", encoding="utf-8") as json_file:
            tokenizer_config = json_file.read()
        tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

        lstm = keras.models.load_model(prediction_data.lstm_model_path + "best_lstm_model.h5")
        vgg16 = keras.models.load_model(prediction_data.vgg16_model_path + "best_vgg16_model.h5")

        with open(prediction_data.model_weights_path + "best_weights.json", "r") as json_file:
            best_weights = json.load(json_file)

        with open(prediction_data.mapper_path + "mapper.json", "r") as json_file:
            mapper = json.load(json_file)

        predictor = Predict(
            tokenizer=tokenizer,
            lstm=lstm,
            vgg16=vgg16,
            best_weights=best_weights,
            mapper=mapper,
            filepath=prediction_data.dataset_path,
            imagepath=prediction_data.images_path,
        )

        # Création de l'instance Predict et exécution de la prédiction
        predictions = predictor.predict()

        # Sauvegarde des prédictions
        #with open("data/preprocessed/predictions.json", "w", encoding="utf-8") as json_file:
        #    json.dump(predictions, json_file, indent=2)
    except IOError as e:
        raise HTTPException(
            status_code=404,
            detail="File not found : " + str(e))
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail="Error : " + str(e))    
    
    return predictions