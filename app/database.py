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
    print(f"Erreur de connexion à la base de données: {e}")
    raise e

# Gestion / connexion à la base de donnée locale
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
   Permet de créer une nouvelle session SQLAlchemy pour une requête,
    et la fermer une fois la requête terminée.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()