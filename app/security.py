from datetime import datetime, timedelta, timezone
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
    """
    Creates a JWT access token encoded with a secret key and a specific algorithm.

    The token is created from a supplied data set and can be configured to expire
    after a certain period of time. If no expiry time is supplied, a default time of 15 minutes is applied.

    Parameters :
        data (dict): A dictionary containing the data to be encoded in the token.
        expires_delta (timedelta, optional): A timedelta object representing the time until the token expires.
        If None, a default expiration of 15 minutes is used.

    Returns :
        str: An encoded JWT token containing the supplied data and an expiration mark.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(db: Session = Depends(database.get_db), token: str = Depends(oauth2_scheme)):
    """
    Checks the user against the JWT token.

    This function decodes the supplied token, extracts the user name (sub) and attempts to retrieve the corresponding user from the database. If the token is invalid, expired, or if the user does not exist in the database, an HTTP 401 exception is raised, indicating that the credentials cannot be validated.

    Args:
        db (Session): The SQLAlchemy database session, automatically injected by FastAPI thanks to the `database.get_db` dependency.
        token (str): The JWT token supplied by the user, automatically injected by FastAPI thanks to the `oauth2_scheme` dependency.

    Returns:
        db_models.User: The user instance retrieved from the database if the token is valid and the user exists.

    Raises:
        HTTPException: An HTTP 401 exception is thrown if the token is invalid, expired, or if no corresponding user is found in the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unable to validate credentials",
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