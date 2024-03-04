from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import database, db_models

load_dotenv()  

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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