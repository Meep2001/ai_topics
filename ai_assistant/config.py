import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GROQ_API_KEY')
API_URL = os.getenv('GROQ_API_URL')
MODEL = os.getenv('GROQ_MODEL')

if not all([API_KEY, API_URL, MODEL]):
    raise ValueError('Missing API configuration!')

