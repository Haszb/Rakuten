from sqlalchemy.orm import Session
import db_models, security, schemas
from fastapi import HTTPException

def authenticate_user(db: Session, username: str, password: str):
    """
    Authentifie un utilisateur via son nom d'utilisateur et son mot de passe.

    Renvoi:
        db_models.User : l'utilisateur authentifié ou False si l'authentification échoue.
    """
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if not user or not user.is_active or not security.verify_password(password, user.hashed_password):
        return False
    return user

def create_user(db: Session, user: schemas.UserCreate):
    """
    Tente de créer un nouvel utilisateur dans la base de données.

    Args:
        db (Session): La session de la base de données.
        user (schemas.UserCreate): Le schéma de l'utilisateur à créer.

    Returns:
        L'instance de l'utilisateur créé.

    Raises:
        HTTPException: Avec un statut 400 pour un conflit de nom d'utilisateur ou un problème de validation.
    """
    # Hachage du mot de passe utilisateur
    hashed_password = security.get_password_hash(user.password)
    db_user = db_models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        is_active=True,
        role=user.role
    )
    db.add(db_user)
    try:
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Ce nom d'utilisateur ou email existe déjà.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur inattendue lors de la création de l'utilisateur: {str(e)}")


def get_user_by_username(db: Session, username: str, raise_exception: bool = True):
    """
    Récupère un utilisateur via son nom d'utilisateur.

    Args :
        db (Session) : La session de la base de données.
        username (str) : Le nom d'utilisateur recherché.

    Renvoi :
        db_models.User : L'utilisateur s'il a été trouvé, sinon aucun.
    """
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if user is None and raise_exception:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé.")
    return user

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

    try:
        db.commit()
        db.refresh(user)
        return user
    except:
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour de l'utilisateur.")

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

    try:
        db.delete(user)
        db.commit()
        return {"detail": "Utilisateur supprimé"}
    except:
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de l'utilisateur.")
