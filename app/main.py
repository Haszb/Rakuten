import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import schemas, crud, database, security, db_models
from security import get_current_user, oauth2_scheme
from schemas import PredictionData, User, UserCreate, PredictionOption
from database import get_db, engine, Base, SessionLocal
from crud import get_user_by_username, create_user  
from datetime import timedelta
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv
from tensorflow import keras
import pandas as pd
import json
from src.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.models.train_model import TextLSTMModel, ImageVGG16Model, concatenate
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import random
import shutil
import csv
import pickle
from src.predict import Predict
from datetime import datetime

# Chargement des variables d'environnement 
load_dotenv()

# Récupération des variables d'environnement 
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


# Structure les routes de l'API en "domaines"   
tags_metadata = [
    {
        "name": "User management",
        "description": "User management: from creation to updating and deletion, not forgetting token management.",
    },
    {
        "name": "Model features",
        "description": "API features such as product category prediction based on descriptions and images, training ...",
    },
    {
        "name": "System",
        "description": "System operations such as checking API operating status",
    },
    {
        "name": "New product",
        "description": "Features for adding new products, predicting new products, feedback and statistics",
    }
]

users_router = APIRouter()

@users_router.post("/users", response_model=schemas.User, tags=["User management"])
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    if current_user.role != db_models.Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized user")
    
    existing_user = db.query(db_models.User).filter(db_models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This username already exist")
    
    return crud.create_user(db=db, user=user)

@users_router.put("/users/{identifier}", response_model=schemas.User, tags=["User management"])
def update_user(identifier: str, user: schemas.UserUpdate, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    if current_user.role != db_models.Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized user")
    db_user = crud.update_user_by_identifier(db=db, identifier=identifier, user_update=user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@users_router.delete("/users/{identifier}", status_code=status.HTTP_204_NO_CONTENT, tags=["User management"])
def delete_user(identifier: str, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    if current_user.role != db_models.Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized user")
    success = crud.delete_user_by_identifier(db=db, identifier=identifier)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"detail": "User deleted"}

# Initialisation de l'API
api = FastAPI(
    title="Rakuten API",
    description="Our RAKUTEN API Project.",
    version="1.2",
    openapi_tags=tags_metadata)

api.include_router(users_router)

@api.on_event("startup")
async def startup_event():
    """
    At startup: 
        - checks if users table exists and creates one if it doesn't 
        - checks for the existence of an admin user and creates one if it doesn't exist   
    """
    db = database.SessionLocal() 
    Base.metadata.create_all(bind=engine)

    try:
        admin_user = crud.get_user_by_username(db, username=ADMIN_USERNAME, raise_exception=False)
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

@api.get("/", response_class=HTMLResponse, tags=["System"])
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
@api.post("/token", tags=["User management"])
async def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@api.get('/ping', tags=["System"])
async def get_ping():
    return "It's working"

@api.post('/prediction', tags=["Model features"])
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

        lstm = keras.models.load_model(prediction_data.lstm_model_path + "best_lstm_model.h5", compile= False)
        vgg16 = keras.models.load_model(prediction_data.vgg16_model_path + "best_vgg16_model.h5", compile= False)

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
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
    #except IOError as e:
    #    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e))    
    
    return predictions

# Generate random ID within a specified range
def generate_productid():
    return random.randint(4252011632, 10000000000)
def generate_imageid():
    return random.randint(1328824385, 10000000000)

@api.post("/new_product", tags=["New product"])
async def create_product(designation: str = Form(...),
                         description: str = Form(...),
                         image: UploadFile = File(...),
                         current_user: schemas.User = Depends(get_current_user)):
    """
    Create a new product with the provided designation, description, and image.

    Parameters:
    - designation (str): The designation of the product.
    - description (str): The description of the product.
    - image (UploadFile): The image file of the product.

    Returns:
    - dict: A dictionary containing the result of the operation, or an error message if the image is invalid.
    """
    # Check if the user has the necessary permissions
    if current_user.role not in [db_models.Role.admin, db_models.Role.employe]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators and employees.")
       
    try:
        # Open and verify the image file
        img = Image.open(image.file)
        img.verify()
        
        # Generate unique identifiers for the product and image
        productid = generate_productid()
        imageid = generate_imageid()
        
        # Create a dictionary representing the new product
        new_product = {
            "designation" : designation,
            "description" : description,
            "productid" : productid,
            "imageid" : imageid
        }
        # Create directories if they don't exist
        if not os.path.exists("../data/new_product"):
            os.makedirs("../data/new_product")
            
        if not os.path.exists("../data/new_product/image"):
            os.makedirs("../data/new_product/image")
            
        # Read or create the CSV file for storing product data            
        if os.path.exists("../data/new_product/new_product.csv"):
            new_product_df = pd.read_csv("../data/new_product/new_product.csv")
        else:
            new_product_df = pd.DataFrame(columns=["designation", "description", "productid", "imageid"])
      
        # Append the new product to the CSV file
        new_product_df.loc[len(new_product_df)] = new_product
        new_product_df.to_csv("../data/new_product/new_product.csv", index= False)

        # Rename and save the image file
        image_name = f"image_{new_product['imageid']}_product_{new_product['productid']}.jpg"
        upload_directory = "../data/new_product/image/"
        image_path = os.path.join(upload_directory, image_name)
        img = Image.open(image.file)
        img.save(image_path)
              
        # Create a CSV file for prediction
        predict_df = pd.DataFrame([new_product])
        predict_df.to_csv("../data/new_product/to_predict.csv")

        return "The database has been updated"
    
    except (IOError, SyntaxError) as e:
        return {"error": "Invalid image file"}

@api.post("/predict_new_product", tags=["New product"])
async def predict_new_product(prediction_data: PredictionData, db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    """
    Return the prediction of the newproduct category.

    Parameters:
    -----------
        prediction_data: PredictionData
            PredictionData object which has the the csv file path for products to categorize and also the images path.
    """
    # Check if the user has the necessary permissions
    if current_user.role not in [db_models.Role.admin, db_models.Role.employe]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators and employees..")
    try:
        # Load configurations and models
        with open(prediction_data.tokenizer_config_path + "tokenizer_config.json", "r", encoding="utf-8") as json_file:
            tokenizer_config = json_file.read()
        tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

        lstm = keras.models.load_model(prediction_data.lstm_model_path + "best_lstm_model.h5", compile=False)
        vgg16 = keras.models.load_model(prediction_data.vgg16_model_path + "best_vgg16_model.h5", compile=False)

        with open(prediction_data.model_weights_path + "best_weights.json", "r") as json_file:
            best_weights = json.load(json_file)

        with open(prediction_data.mapper_path + "mapper.json", "r") as json_file:
            mapper = json.load(json_file)

        prediction_data.dataset_path = "../data/new_product/to_predict.csv"
        prediction_data.images_path = "../data/new_product/image"
        
        predictor = Predict(
            tokenizer=tokenizer,
            lstm=lstm,
            vgg16=vgg16,
            best_weights=best_weights,
            mapper=mapper,
            filepath=prediction_data.dataset_path,
            imagepath=prediction_data.images_path,
        )

        # Create Predict instance and execute prediction
        predictions = predictor.predict()

        # Save predictions
        with open("../data/new_product/predictions.json", "w", encoding="utf-8") as json_file:
            json.dump(predictions, json_file, indent=2)
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
    #except IOError as e:
    #    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e))    
    
    return predictions

# Check if the predictions file exists
predictions_file = "../data/new_product/predictions.json"
if os.path.exists(predictions_file):
    # If it exists, open the file and load the prediction data
    with open(predictions_file, "r") as pred_file:
        prediction_data = json.load(pred_file)
        # Extract the model prediction from the prediction data        
        model_prediction = prediction_data.get("0", None)
else:
    # If the predictions file does not exist, set the model prediction to 0    
    model_prediction = 0

@api.post("/check_prediction", tags=["New product"])
async def check_prediction(model_prediction: int = model_prediction,
                           verification_prediction: PredictionOption = PredictionOption.success,
                           current_user: schemas.User = Depends(get_current_user)):
    """
    Check the prediction of a new product and save the verified prediction.

    Parameters:
    - model_prediction (int): The prediction made by the model.
    - verification_prediction (PredictionOption): The verification status of the prediction.

    Returns:
    - dict: A dictionary containing the details of the verified prediction.
    """
    # Check if the model prediction is 0, indicating that the process needs to be restarted      
    if model_prediction == 0:
        return "The process needs to be restarted from the beginning or the page needs to be reloaded, thank you."
    
    # Check if the user has the necessary permissions        
    if current_user.role not in [db_models.Role.admin, db_models.Role.employe]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators and employees.")
    try:
        # Check if the prediction verified CSV file exists, otherwise create it        
        if os.path.exists("../data/new_product/prediction_verified.csv"):
            product_pred_verified = pd.read_csv("../data/new_product/prediction_verified.csv", index_col=0)
        else:
            product_pred_verified = pd.DataFrame(columns=["date", "user",
                                                            "designation", "description", "productid", "imageid",
                                                            "model_prediction", "verified_prediction", "Result"])
        
        # Get the current date and the username of the current user        
        prediction_date = datetime.today().strftime("%Y-%m-%d")
        user = current_user.username
        # Read the to_predict CSV file to get the prediction data        
        to_predict_df = pd.read_csv("../data/new_product/to_predict.csv")
        
        # Determine if the verification process was successful        
        verification_process = (verification_prediction == PredictionOption.success) | (verification_prediction == str(model_prediction))
        new_row = {
            "date": prediction_date,
            "user": user,
            "designation": str(to_predict_df["designation"].iloc[0]),
            "description": str(to_predict_df["description"].iloc[0]),
            "productid": int(to_predict_df["productid"].iloc[0]),
            "imageid": int(to_predict_df["imageid"].iloc[0]),
            "model_prediction": int(model_prediction),
            "verified_prediction": int(model_prediction if  verification_process else verification_prediction.value),
            "Result" : str("Success" if verification_process else "Failure")
            }

        # Add the new row to the prediction verified DataFrame and save it to a CSV file        
        new_row_df = pd.DataFrame([new_row])
        product_pred_verified.loc[len(product_pred_verified)] = new_row_df.loc[0]
        product_pred_verified.to_csv("../data/new_product/prediction_verified.csv")
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
    #except IOError as e:
    #    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e))
    
    return new_row

   
@api.post('/Stats', tags=["New product"])
async def get_stats(db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    """
    Retrieves statistics on new product predictions.

    Returns:
    - Dictionary containing the number of new products and calculated accuracy.
    """
    
    # Check if the user has the necessary permissions
    if current_user.role not in [db_models.Role.admin, db_models.Role.employe]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators and employees..")
    
    try:
        
        # Check if the prediction data exist, return a dictionnary if not
        prediction_verified_file = "../data/new_product/prediction_verified.csv"
        if not os.path.exists(prediction_verified_file):
            new_prod_data = {
                "Number of new products" : 0,
                "Calculated accuracy of new product (%)" : 100
            }
            
        else:
            # Load the prediction data  
            prediction_verified_df = pd.read_csv(prediction_verified_file)
            
            #  Check that the lenght of the data is different from 0, to prevent an error (divide by 0)
            if len(prediction_verified_df) == 0:
                new_prod_data = {
                "Number of new products" : 0,
                "Calculated accuracy of new product (%)" : 100
                }               
   
            else:          
                #Calculate  some statistics    
                accuracy_new_product = len(prediction_verified_df[prediction_verified_df['Result'] == "Success"]) / len(prediction_verified_df) *100 
                new_prod_data = {
                    "Number of new products" : len(prediction_verified_df),
                    "Calculated accuracy of new product (%)" : accuracy_new_product
                    }
            # Convert new_prod_data dictionary to JSON format
            new_prod_json = json.dumps(new_prod_data)

            # Save the JSON data to a file
            with open('../data/new_product/new_prod_data.json', 'w') as json_file:
                json_file.write(new_prod_json)
        
        return new_prod_data
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
    #except IOError as e:
    #    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e)) 

@api.post("/move_new_product", tags=['New product'])
async def move_new_product(db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    """
    Moves new product data and images, updates datasets, and archives files.
    And delete all data from the datasets, but keep the files architecture.
    """
    # Check if the user has the necessary permissions
    if current_user.role not in [db_models.Role.admin, db_models.Role.employe]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators and employees..")
    
    try:
        # Check if the prediction_verified.csv file exists   
        if os.path.exists("../data/new_product/prediction_verified.csv"):
            new_product_df = pd.read_csv("../data/new_product/prediction_verified.csv")
        else:
            return "The file prediction_verified.csv does not exist."
        
        # Split the new product data into train and test sets
        sample_size_train = int(len(new_product_df) * 0.8)
        columns_to_keep = ["designation", "description", "productid", "imageid", "verified_prediction"]
        new_prod_to_train = new_product_df[columns_to_keep].sample(sample_size_train, random_state=42)
        new_prod_to_test = new_product_df[columns_to_keep].drop(new_prod_to_train.index)
        
        # Create directories for train and test images
        train_dir = "../data/preprocessed/image_train"
        test_dir = "../data/preprocessed/image_test"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Move images to train or test directories based on the split
        # Iterate through each row in the DataFrame     
        for index, row in new_product_df.iterrows():
            # Generate the image name based on the imageid and productid
            image_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"
            # Determine the destination directory based on whether the row is in new_prod_to_train or new_prod_to_test
            if index in new_prod_to_train.index:
                destination = os.path.join(train_dir, image_name)
                new_product_df.loc[index, "Sent_to"] = "Train"
            elif index in new_prod_to_test.index:
                destination = os.path.join(test_dir, image_name)
                new_product_df.loc[index, "Sent_to"] = "Test"
            else:
                # Skip to the next iteration if the row index is not found in either new_prod_to_train or new_prod_to_test
                continue
            # Construct the source path of the image
            source = os.path.join("../data/new_product/image", image_name)
            # Move the image file from the source directory to the destination directory
            shutil.move(source, destination)


        # Update X_train_update, X_test_update, and Y_train_CVw08PX datasets
        x_train_update = pd.read_csv('../data/preprocessed/X_train_update.csv', index_col=0)
        x_test_update = pd.read_csv("../data/preprocessed/X_test_update.csv", index_col=0)
        y_train_CVw08PX = pd.read_csv("../data/preprocessed/Y_train_CVw08PX.csv", index_col=0)
        
        x_train_update = pd.concat([x_train_update, new_prod_to_train.drop(columns=['verified_prediction'])],
                                ignore_index=True)
        x_train_update.to_csv('../data/preprocessed/X_train_update_with_new_prod.csv')
        
        x_test_update = pd.concat([x_test_update, new_prod_to_test.drop(columns=['verified_prediction'])],
                                ignore_index=True)
        x_test_update.to_csv('../data/preprocessed/X_test_update_with_new_prod.csv')
        
        product_code = new_prod_to_train.rename(columns= {'verified_prediction':'prdtypecode'}).drop(columns=["designation","description","productid","imageid"])
        y_train_CVw08PX = pd.concat([y_train_CVw08PX, product_code],
                                    ignore_index=True)
        y_train_CVw08PX.to_csv('../data/preprocessed/y_train_update.csv')
        
        # Archive new_product data in a file named with the current date
        archive_dir = "../data/archive/new_product"
        os.makedirs(archive_dir, exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        # This line is added in case of the function going twice a day, so it don't overwrite the file.
        current_hour = datetime.now().strftime("%H-%M-%S")
        destination_dir = os.path.join(f"{archive_dir}/{current_date}_{current_hour}.csv")
        
        # Check if the file already exist :          
        new_product_df.to_csv(destination_dir)
        
        # Delete all data but columns in new_product's csv to start anew.
        source_dir = "../data/new_product"
        for file in os.listdir(source_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(source_dir, file)
                # Read only the header (i.e., column names) of the CSV file
                with open(file_path, 'r', newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                # Rewrite the CSV file with just the header
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
    #except IOError as e:
    #    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e)) 
    
@api.post("/retrain", tags=['Model features'])
async def train_model(db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    """
    Train and archive machine learning models.
    """    # Check if the user has the necessary permissions
    if current_user.role not in [db_models.Role.admin]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators")
    
    try:
        # Create a folder and move the previous model to it
        work_dir = "../models"
        old_model_dir = "../models/old"
        os.makedirs(old_model_dir, exist_ok=True)
        
        for file_name in os.listdir(work_dir):
            if file_name not in ["mapper.json", "mapper.pkl", "__init__.py"]:
                source_dir = os.path.join(work_dir, file_name)
                shutil.move(source_dir, old_model_dir)
            
        
        # Re-train the model with the updated dataset.
        data_importer = DataImporter()
        df = data_importer.load_data()
        X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

        # Preprocess text and images
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor()
        text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
        text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X_train)
        image_preprocessor.preprocess_images_in_df(X_val)

        # Train LSTM model
        print("Training LSTM Model")
        text_lstm_model = TextLSTMModel()
        text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
        print("Finished training LSTM")

        print("Training VGG")
        # Train VGG16 model
        image_vgg16_model = ImageVGG16Model()
        image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
        print("Finished training VGG")

        with open("../models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
            tokenizer_config = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
        lstm = keras.models.load_model("../models/best_lstm_model.h5")
        vgg16 = keras.models.load_model("../models/best_vgg16_model.h5")

        print("Training the concatenate model")
        model_concatenate = concatenate(tokenizer, lstm, vgg16)
        lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
        best_weights, accuracy = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
        print("Finished training concatenate model")

        with open("../models/best_weights.pkl", "wb") as file:
            pickle.dump(best_weights, file)

        num_classes = 27

        proba_lstm = keras.layers.Input(shape=(num_classes,))
        proba_vgg16 = keras.layers.Input(shape=(num_classes,))

        weighted_proba = keras.layers.Lambda(
            lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
        )([proba_lstm, proba_vgg16])

        concatenate_model = keras.models.Model(
            inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
        )
        # Save the model in h5 format
        concatenate_model.save("../models/concatenate.h5")
        print('Model saved')

        # Save the value of the accuracy used to obtain de best_weights
        with open("../models/accuracy.json", "w") as json_file:
            json.dump({"accuracy": accuracy}, json_file)
        print("Accuracy saved")
        
        # Create an archive of the model
        for file_name in os.listdir(work_dir):
            date_actuelle = datetime.now().strftime("%Y-%m-%d")
            archive_dir = os.path.join("../data/archive/model", date_actuelle)
            if os.path.isfile(work_dir) and file_name not in ["mapper.json", "mapper.pkl", "__init__.py"]:
                shutil.copytree(work_dir, archive_dir)
        print("Model archived")
                
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
    #except IOError as e:
    #    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e)) 

@api.post("/validation", tags=['Model features'])
async def compare_models(db: Session = Depends(database.get_db), current_user: schemas.User = Depends(get_current_user)):
    """
    Return current model's accuracy
    """
    if current_user.role not in [db_models.Role.admin]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Operation only allowed for administrators")

    try:
        with open("../models/accuracy.json", "r") as json_file:
            new_accuracy = json.load(json_file)

        return new_accuracy['accuracy']
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found : " + str(e))
#except IOError as e:
#    raise HTTPException(status_code=404, detail="IO Error : " + str(e))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error : " + str(e)) 