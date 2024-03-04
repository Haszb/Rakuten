from sqlalchemy import Column, Integer, String, Boolean, Enum
from sqlalchemy.orm import validates
from database import Base
from enum import Enum as PyEnum

class Role(PyEnum):
    """
    Enum rôles utilisateurs disponibles
    """
    admin = "Admin"
    employe = "Employe"
    client = "Client"

class User(Base):
    """
    Modèle représentant un utilisateur au sein de la base de données. 

    Attributs:
        id (int): clé primaire
        username (str): nom d'utilisateur unique pour l'utilisateur
        email (str): e-mail unique pour l'utilisateur
        hashed_password (str): mot de passe hashé
        is_active (bool): statut de l'utilisateur
        role (Enum[Role]): rôle de l'utilisateur
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(Enum(Role))

    @validates('email')
    def validate_email(self, key, email):
        assert '@' in email, "l'e-mail doit contenir @"
        return email

    @validates('username')
    def validate_username(self, key, username):
        assert len(username) >= 4, "Le nom d'utilisateur doit contenir au moins 4 caractères"
        return username