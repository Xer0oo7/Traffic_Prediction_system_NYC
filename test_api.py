
import requests
import os
from dotenv import load_dotenv

class GoogleMapsAPITester:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/directions/json"

    def test_route(self, origin, destination):
        params = {
            'origin': origin,
            'destination': destination,
            'key': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data.get('status', 'No status')}")
            if data.get('status') == 'OK':
                route = data['routes'][0]['legs'][0]
                print(f"Distance: {route['distance']['text']}")
                print(f"Duration: {route['duration']['text']}")
            else:
                print(f"API Error: {data.get('status')}")
        else:
            print(f"HTTP Error: {response.status_code}")

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("Please set GOOGLE_MAPS_API_KEY in your .env file")
    else:
        tester = GoogleMapsAPITester(api_key)
        tester.test_route('New York, NY', 'Brooklyn, NY')
