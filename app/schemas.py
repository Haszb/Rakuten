from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from db_models import Role
from enum import Enum

class UserCreate(BaseModel):
    """
    Schema for user creation
    """
    username: str = Field(..., example="DataScientest", min_length=4, description="Unique user name")
    email: EmailStr = Field(..., example="admin@datascientest.com", description="Unique user e-mail")
    password: str = Field(..., example="nepasmettre123456", min_length=6, description="User's password")
    role: Role = Field(..., description="Rôle attribué à l'utilisateur")

class User(BaseModel):
    """
    Response scheme for obtaining user information
    """
    id: int = Field(..., example=1, description="User's unique ID")
    is_active: bool = Field(..., example=True, description="User's status : active / inactive")
    username: str = Field(..., example="DataScientest", description="User name")
    email: EmailStr = Field(..., example="admin@datascientest.com", description="User's e-mail")
    role: Role = Field(..., description="User's role")
    
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    """
    Schema for user modification
    """
    username: Optional[str] = Field(None, description="New username")
    email: Optional[EmailStr] = Field(None, description="New e-mail")
    is_active: Optional[bool] = Field(None, description="Enable or disable user account")
    role: Optional[Role] = Field(None, description="New user's role")  

class PredictionData(BaseModel):
    """
    A class to represent a product.
    
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


class PredictionOption(str, Enum):
    """
    Enumerates the different prediction options.

    Attributes:
        success: Indicates that the prediction was successful.
        optionX: Represents different classes of predictions.
    """
    success = "Success"
    option1 = "10"
    option2 = "2280"
    option3 = "50"
    option4 = "1280"
    option5 = "2705"
    option6 = "2522"
    option7 = "2582"
    option8 = "1560"
    option9 = "1281"
    option10 = "1920"
    option11 = "2403"
    option12 = "1140"
    option13 = "2583"
    option14 = "1180"
    option15 = "1300"
    option16 = "2462"
    option17 = "1160"
    option18 = "2060"
    option19 = "40"
    option20 = "60"
    option21 = "1320"
    option22 = "1302"
    option23 = "2220"
    option24 = "2905"
    option25 = "2585"
    option26 = "1940"
    