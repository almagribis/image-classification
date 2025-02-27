from pydantic.v1 import BaseSettings
from dotenv import load_dotenv
import os

class AppConfig(BaseSettings):
    """setup config"""
    MODEL_SVC_PATH : str
    MODEL_CNN_PATH : str
    
    class Config:
        env_file = ".env"
    load_dotenv(os.path.join("..",".env"), override=True)
settings = AppConfig()