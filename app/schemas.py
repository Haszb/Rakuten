from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from db_models import Role

class UserCreate(BaseModel):
    """
    Schema pour la création d'un utilisateur
    """
    username: str = Field(..., example="DataScientest", min_length=4, description="Nom d'utilisateur unique pour l'utilisateur")
    email: EmailStr = Field(..., example="admin@datascientest.com", description="E-mail unique pour l'utilisateur")
    password: str = Field(..., example="nepasmettre123456", min_length=6, description="Mot de passe de l'utilisateur")
    role: Role = Field(..., description="Rôle attribué à l'utilisateur")

class User(BaseModel):
    """
    Schema de réponse pour obtenir les informations sur un utilisateur
    """
    id: int = Field(..., example=1, description="l'ID unique de l'utilisateur")
    is_active: bool = Field(..., example=True, description="Statut de l'utilisateur : actif / inactif")
    username: str = Field(..., example="DataScientest", description="Le nom d'utilisateur.")
    email: EmailStr = Field(..., example="admin@datascientest.com", description="Le mail attaché à l'utilisateur")
    role: Role = Field(..., description="Le rôle de l'utilisateur")
    
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    """
    Schema modifier les informations d'un utilisateur
    """
    username: Optional[str] = Field(None, description="Le nouveau nom d'utilisateur")
    email: Optional[EmailStr] = Field(None, description="Le nouvel e-mail")
    is_active: Optional[bool] = Field(None, description="Active ou désactive le compte utilisateur")
    role: Optional[Role] = Field(None, description="Le nouveau rôle utilisateur")  

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
    tokenizer_config_path:Optional[str] = "../models/" 
    lstm_model_path:Optional[str] = "../models/"
    vgg16_model_path:Optional[str] = "../models/" 
    model_weights_path:Optional[str] = "../models/"
    mapper_path:Optional[str] = "../models/"
