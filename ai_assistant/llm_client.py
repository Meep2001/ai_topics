import requests
import json
from config import API_KEY, API_URL, MODEL

def get_response(message):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-type': 'application/json'
    }
    payload = {
        'model': MODEL,
        'messages': message,
        'temperature': 0.7,
        "max_completion_tokens": 150
    }
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = json.loads(response.text)
        message = result['choices'][0]['message']['content']
        return message
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.HTTPError as e:
        return f"HTTP error:{e}"
    except requests.exceptions.HTTPError as e:
        return f"Request Error:{e}"
    except(KeyError,ValueError,IndexError):
        return f"Error: Unexpected API response"
    