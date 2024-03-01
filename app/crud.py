from sqlalchemy.orm import Session
from . import db_models, security, schemas
from fastapi import HTTPException

def authenticate_user(db: Session, username: str, password: str):
    """
    Authentifie un utilisateur via son nom d'utilisateur et de son mot de passe.

    Renvoi:
        db_models.User : l'utilisateur authentifié ou False si l'authentification échoue.
    """
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if not user or not user.is_active or not security.verify_password(password, user.hashed_password):
        return False
    return user

def create_user(db: Session, user: schemas.UserCreate):
    """
     Crée un nouvel utilisateur dans la base de données.

    Args :
        db (Session) : La session de la base de données.
        user (schemas.UserCreate) : Les détails de l'utilisateur, cf. schemas.py

    Renvoi :
        db_models.User : L'utilisateur créé.
    """
    try:
        hashed_password = security.get_password_hash(user.password)
        db_user = db_models.User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            is_active=True,  
            role=user.role
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur lors de la création de l'utilisateur.")

def get_user_by_username(db: Session, username: str):
    """
    Récupère un utilisateur via son nom d'utilisateur.

    Args :
        db (Session) : La session de la base de données.
        username (str) : Le nom d'utilisateur recherché.

    Renvoi :
        db_models.User : L'utilisateur s'il a été trouvé, sinon aucun.
    """
    try:
        user = db.query(db_models.User).filter(db_models.User.username == username).first()
        if user:
            return user
        else:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de l'utilisateur.")

def update_user_by_identifier(db: Session, identifier: str, user_update: schemas.UserUpdate):
    """
    Met à jour un utilisateur basé sur l'ID, l'email ou le nom d'utilisateur.

    Args:
        db (Session): Session de base de données.
        identifier (str): Peut être l'ID, l'email ou le nom d'utilisateur.
        user_update (schemas.UserUpdate): Contient les champs à mettre à jour.
    """
    if identifier.isdigit():  
        user = db.query(db_models.User).filter(db_models.User.id == int(identifier)).first()
    else:
        user = db.query(db_models.User).filter((db_models.User.username == identifier) | (db_models.User.email == identifier)).first()

    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    update_data = user_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return user

def delete_user_by_identifier(db: Session, identifier: str):
    """
    Supprime un utilisateur à partir de l'ID, l'e-mail ou le nom d'utilisateur.

    Args:
        db (Session): Session de base de données.
        identifier (str): soit l'ID soit l'e-mail soit le nom d'utilisateur.
    """
    if identifier.isdigit():
        user = db.query(db_models.User).filter(db_models.User.id == int(identifier)).first()
    else:
        user = db.query(db_models.User).filter((db_models.User.username == identifier) | (db_models.User.email == identifier)).first()

    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    db.delete(user)
    db.commit()
    return {"detail": "Utilisateur supprimé"}

