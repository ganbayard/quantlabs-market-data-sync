import os
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
import streamlit as st

# Database configuration
class DatabaseConfig:
    def __init__(self, db_type="sqlite"):
        self.db_type = db_type
        self.engine = None
        self.session = None
        self.Base = declarative_base()
        
    def get_connection_string(self):
        if self.db_type == "sqlite":
            # SQLite connection
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "finance.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite:///{db_path}"
        elif self.db_type == "mysql":
            # MySQL connection
            user = st.secrets.get("db_user", "root")
            password = st.secrets.get("db_password", "root")
            host = st.secrets.get("db_host", "localhost")
            port = st.secrets.get("db_port", "3306")
            database = st.secrets.get("db_name", "finance")
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def connect(self):
        connection_string = self.get_connection_string()
        self.engine = create_engine(connection_string, echo=False)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        return self.engine, self.session
    
    def create_tables(self, Base):
        Base.metadata.create_all(self.engine)
    
    def close(self):
        if self.session:
            self.session.close()

# Initialize database connection
def get_db_connection(db_type="sqlite"):
    db_config = DatabaseConfig(db_type)
    engine, session = db_config.connect()
    return db_config, engine, session
