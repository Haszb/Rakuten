from fastapi import FastAPI, Depends, HTTPException, status, APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from . import schemas, crud, database, security, db_models
from .schemas import PredictionData, User, UserCreate
from .database import get_db, engine, Base, SessionLocal
from .security import ALGORITHM
from .crud import get_user_by_username, create_user  
from datetime import timedelta
from jose import jwt, JWTError
from typing import Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
import os
from tensorflow import keras
import pandas as pd
import json
from src.features.build_features import TextPreprocessor, ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import Predict


# Chargement des variables d'environnement 
load_dotenv()

# Récupération des variables d'environnement 
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Schema OAuth2 avec les jetons 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Structure les routes de l'API en "domaines"   
tags_metadata = [
    {
        "name": "Utilisateurs",
        "description": "Gestion des utilisateurs : de la création à la suppression en passant par la mise à jour, sans oublié la gestion des tokens.",
    },
    {
        "name": "Fonctionnalités",
        "description": "Fonctionnalités de l'API comme prédiction de catégorie de produits basée sur les descriptions et les images, l'entrainement ...",
    },
    {
        "name": "Système",
        "description": "Opérations système comme par exemple la vérification de l'état de l'API.",
    }
]

async def get_current_user(db: Session = Depends(database.get_db), token: str = Depends(oauth2_scheme)):
    """
    Vérifie l'utilisateur à partir du token JWT.

    Cette fonction décode le token fourni, extrait le nom d'utilisateur (sub) et tente de récupérer l'utilisateur correspondant dans la base de données. Si le token est invalide, expiré, ou si l'utilisateur n'existe pas dans la base de données, une exception HTTP 401 est levée, indiquant que les credentials ne peuvent pas être validés.

    Args:
        db (Session): La session de base de données SQLAlchemy, injectée automatiquement par FastAPI grâce à la dépendance `database.get_db`.
        token (str): Le token JWT fourni par l'utilisateur, injecté automatiquement par FastAPI grâce à la dépendance `oauth2_scheme`.

    Returns:
        db_models.User: L'instance de l'utilisateur récupérée de la base de données si le token est valide et que l'utilisateur existe.

    Raises:
        HTTPException: Une exception HTTP 401 est levée si le token est invalide, expiré, ou si aucun utilisateur correspondant n'est trouvé dans la base de données.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les identifiants",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

users_router = APIRouter()

@users_router.post("/users", response_model=schemas.User, tags=["Utilisateurs"])
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    if current_user.role != db_models.Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Utilisateur non autorisé")
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ce nom d'utilisateur existe déjà")
    return crud.create_user(db=db, user=user)

@users_router.put("/users/{identifier}", response_model=schemas.User, tags=["Utilisateurs"])
def update_user(identifier: str, user: schemas.UserUpdate, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    if current_user.role != db_models.Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Utilisateur non autorisé")
    db_user = crud.update_user_by_identifier(db=db, identifier=identifier, user_update=user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return db_user

@users_router.delete("/users/{identifier}", status_code=status.HTTP_204_NO_CONTENT, tags=["Utilisateurs"])
def delete_user(identifier: str, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    if current_user.role != db_models.Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Utilisateur non autorisé")
    success = crud.delete_user_by_identifier(db=db, identifier=identifier)
    if not success:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return {"detail": "Utilisateur supprimé"}

# Initialisation de l'API
api = FastAPI(
    title="Rakuten API",
    description="Our RAKUTEN API Project.",
    version="1.1",
    openapi_tags=tags_metadata)

api.include_router(users_router)

@api.on_event("startup")
async def startup_event():
    """
    Au démarrage : 
    - Vérifie si la table users existe et la crée si ce n'est pas le cas 
    - vérifie l'existence d'un utilisateur admin  et en crée un s'il n'existe pas   
    """
    db = database.SessionLocal() 
    Base.metadata.create_all(bind=engine)

    try:
        admin_user = crud.get_user_by_username(db, username=ADMIN_USERNAME)
        if not admin_user:
            user_in = schemas.UserCreate(
                username=ADMIN_USERNAME, 
                email=ADMIN_EMAIL, 
                password=ADMIN_PASSWORD,
                role=db_models.Role.admin
            )
            crud.create_user(db=db, user=user_in)
        else:
            print("Admin user already exists.")
    except OperationalError as e:
        print(f"Database access error: {e}")
    finally:
        db.close()

@api.get("/", response_class=HTMLResponse, tags=["Système"])
def home():
    """
    Renvoie une page d'index avec des liens vers la documentation de l'API et la spécification OpenAPI.
    """
    return """
    <html>
        <head>
            <title>Projet RAKUTEN API - Index</title>
        </head>
        <body>
            <h1>Bienvenue sur notre API de classification des produits Rakuten</h1>
            <p>Ce projet vise à développer et déployer une application de classification de produits pour Rakuten, en utilisant des approches basées sur le traitement du langage naturel pour les descriptions textuelles des produits et la computer vision pour les images des produits. L'application doit classifier automatiquement les produits du catalogue de Rakuten afin d’éviter un classement manuel des produits par leur équipes. </p>
            <h2>Utilisez les liens ci-dessous pour accéder à la documentation de l'API :</h2>
            <ul>
                <li><a href="/docs">Swagger UI</a></li>
                <li><a href="/redoc">Redoc</a></li>
                <li><a href="/openapi.json">Spécification OpenAPI</a></li>
            </ul>
            <h3>Projet MLOps NOV23 - DataScientest</h3>
            <ul>
                <li>BEAUVA Christophe</li>
                <li>de PERETTI Gilles</li>
                <li>SIMO Eric</li>
                <li>ZBIB Hassan</li>
            </ul>

        </body>
    </html>
    """
@api.post("/token", tags=["Utilisateurs"])
async def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@api.get('/ping', tags=["Système"])
async def get_ping():
    return "It's working"

@api.post('/prediction', tags=["Fonctionnalités"])
async def get_prediction(prediction_data: PredictionData, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    """
    Return the prediction of the product category.

    Parameters:
    -----------
        prediction_data: PredictionData
            PredictionData object which has the the excel file path for products to categorize and also the images path.
    """
     # Vérifie si l'utilisateur est admin ou employe
    if current_user.role not in [db_models.Role.admin, db_models.Role.employe]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès refusé. Opération autorisée uniquement pour les administrateurs et les employés.")
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