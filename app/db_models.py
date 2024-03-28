from sqlalchemy import Column, Integer, String, Boolean, Enum
from sqlalchemy.orm import validates
from database import Base
from enum import Enum as PyEnum

class Role(PyEnum):
    """
    Enum available user roles
    """
    admin = "Admin"
    employe = "Employe"
    client = "Client"

class User(Base):
    """
    Model representing a user within the database. 

    Args:
        id (int): primary key
        username (str): unique username for the user
        email (str): user's unique e-mail address
        hashed_password (str): hashed password
        is_active (bool): user status
        role (Enum[Role]): user's role
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
        assert '@' in email, "Email must contain @"
        return email

    @validates('username')
    def validate_username(self, key, username):
        assert len(username) >= 4, "Username must contain at least 4 characters"
        return username