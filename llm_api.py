import os
import requests 
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ['LLM_API_KEY']
BASE_URL = os.environ['LLM_BASE_URL']

url = f'{BASE_URL}/chat/completions'

header = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

payload = {
    'model': 'gpt-5.5',
    'messages': [
        { 'role': 'system', 'content': 'You are helpful AI assistant'},
        { 'role': 'user', 'content': 'explain rest api\'s in simple terms' }
    ],
    'temperature': 0.5
}

responses = requests.post(
    url,
    headers=header,
    json=payload,
    timeout=30
)

responses.raise_for_status()
data = responses.json()
print(data)