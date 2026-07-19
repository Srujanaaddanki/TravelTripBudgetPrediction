import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("ORS_API_KEY")

headers = {
    "Authorization": key
}

url = (
    "https://api.openrouteservice.org/v2/directions/driving-car"
)

body = {
    "coordinates": [
        [79.4192, 28.6139],   # Delhi
        [79.0669, 30.7346]    # Kedarnath approx
    ]
}

response = requests.post(
    url,
    json=body,
    headers=headers
)

print(response.json())