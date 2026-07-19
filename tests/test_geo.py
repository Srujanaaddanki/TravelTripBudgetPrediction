import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEOAPIFY_API_KEY")

destination = input("Enter destination: ")

url = (
    f"https://api.geoapify.com/v1/geocode/search"
    f"?text={destination}"
    f"&apiKey={key}"
)

response = requests.get(url)

print(response.json())