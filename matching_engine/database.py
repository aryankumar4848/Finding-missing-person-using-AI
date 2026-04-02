import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class User(Base):
    """
    Represents an enrolled identity in the system.
    Stores NO biometric data or images, only standard relational metadata.
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Store dynamic generic properties (e.g. employee_id, clearance_level)
    metadata_json = Column(JSON, nullable=True)
    
    # One-to-Many relationship with Hashes
    hashes = relationship("BiometricHash", back_populates="user", cascade="all, delete-orphan")


class BiometricHash(Base):
    """
    Stores the securely transformed Region-Based BioHashes.
    Irreversible via sign() quantization.
    Raw image data is strictly forbidden in this table.
    """
    __tablename__ = 'hashes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Example JSON payload: 
    # {"left_eye": "1001011...", "right_eye": "11100...", "nose":...}
    # Using JSON allows dialect-agnostic storage compatible with PostgreSQL JSONB
    region_hashes = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="hashes")


class DatabaseManager:
    """
    Manages connections and schema initializations.
    Designed for SQLite initially, entirely compatible with PostgreSQL.
    """
    def __init__(self, db_url: str = "sqlite:///biometrics.db"):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def init_db(self):
        """Creates all tables if they don't exist yet."""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Returns a new ORM session."""
        return self.SessionLocal()
