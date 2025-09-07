# Traffic Prediction Prototype Setup Guide

## 1. Clone the repository

## 2. Create a virtual environment

```
python -m venv traffic_prediction_env
```

## 3. Activate the virtual environment

- On Windows:
```
traffic_prediction_env\Scripts\activate
```
- On macOS/Linux:
```
source traffic_prediction_env/bin/activate
```

## 4. Install dependencies

```
pip install -r requirements.txt
```

## 5. Add your Google Maps API key

Edit the `.env` file and paste your API key:
```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

## 6. Run the prototype

```
python traffic_prediction_prototype.py
```
