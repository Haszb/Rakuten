from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"

# Création du moteur SQLAlchemy.
try:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=True  
    )
except SQLAlchemyError as e:
    print(f"Database connection error: {e}")
    raise e

# Gestion / connexion à la base de donnée locale
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Creates a new SQLAlchemy session for a query,
    and close it once the request has been completed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()