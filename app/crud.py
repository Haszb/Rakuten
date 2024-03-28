from sqlalchemy.orm import Session
import db_models, security, schemas
from fastapi import HTTPException

def authenticate_user(db: Session, username: str, password: str):
    """
    Authenticates a user via username and password.

    Returns:
        db_models.User : the authenticated user or False if authentication fails.
    """
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if not user or not user.is_active or not security.verify_password(password, user.hashed_password):
        return False
    return user

def create_user(db: Session, user: schemas.UserCreate):
    """
    Attempts to create a new user in the database.

    Args:
        db (Session): The database session.
        user (schemas.UserCreate): The schema of the user to be created.

    Returns:
        The instance of the created user.

    Raises:
        HTTPException: With status 400 for a username conflict or validation problem.
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
        raise HTTPException(status_code=400, detail="This e-mail or username already exist")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error when creating user.: {str(e)}")


def get_user_by_username(db: Session, username: str, raise_exception: bool = True):
    """
    Retrieves a user by username.

    Args :
        db (Session): The database session.
        username (str): The searched username.

    Return :
        db_models.User: The user if found, otherwise none.
    """
    user = db.query(db_models.User).filter(db_models.User.username == username).first()
    if user is None and raise_exception:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def update_user_by_identifier(db: Session, identifier: str, user_update: schemas.UserUpdate):
    """
    Updates a user based on ID, email or username.

    Args:
        db (Session): Database session.
        identifier (str): Can be ID, email or username.
        user_update (schemas.UserUpdate): Contains fields to be updated.
    """
    if identifier.isdigit():  
        user = db.query(db_models.User).filter(db_models.User.id == int(identifier)).first()
    else:
        user = db.query(db_models.User).filter((db_models.User.username == identifier) | (db_models.User.email == identifier)).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = user_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    try:
        db.commit()
        db.refresh(user)
        return user
    except:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error when updating user.")

def delete_user_by_identifier(db: Session, identifier: str):
    """
    Deletes a user by ID, e-mail or username.

    Args:
        db (Session): Database session.
        identifier (str): either ID, e-mail or username.
    """
    if identifier.isdigit():
        user = db.query(db_models.User).filter(db_models.User.id == int(identifier)).first()
    else:
        user = db.query(db_models.User).filter((db_models.User.username == identifier) | (db_models.User.email == identifier)).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        db.delete(user)
        db.commit()
        return {"detail": "User deleted"}
    except:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error when deleting user.")
