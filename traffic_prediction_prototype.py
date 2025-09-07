
"""
Traffic Prediction Prototype using Google Maps API
Simple implementation for getting started with traffic prediction
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GoogleMapsTrafficCollector:
    """
    Simple class to collect traffic data from Google Maps API
    """
    def __init__(self, api_key):
        """
        Initialize with Google Maps API key

        Args:
            api_key (str): Your Google Maps API key
        """
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/directions/json"

    def get_traffic_data(self, origin, destination, departure_time=None):
        """
        Get current traffic data between two locations

        Args:
            origin (str): Starting location (address or coordinates)
            destination (str): Ending location (address or coordinates) 
            departure_time (datetime): When to leave (optional)

        Returns:
            dict: Traffic data including duration, distance, and traffic info
        """
        # Set departure time to now if not specified
        if departure_time is None:
            departure_time = datetime.now()

        # Convert to Unix timestamp
        departure_timestamp = int(departure_time.timestamp())

        # API parameters
        params = {
            'origin': origin,
            'destination': destination,
            'departure_time': departure_timestamp,
            'traffic_model': 'best_guess',
            'key': self.api_key
        }

        try:
            # Make API request
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if data['status'] == 'OK':
                route = data['routes'][0]['legs'][0]

                # Extract traffic information
                return {
                    'timestamp': departure_time,
                    'origin': origin,
                    'destination': destination,
                    'distance_meters': route['distance']['value'],
                    'distance_text': route['distance']['text'],
                    'duration_seconds': route['duration']['value'],
                    'duration_text': route['duration']['text'],
                    'duration_in_traffic_seconds': route.get('duration_in_traffic', {}).get('value', None),
                    'duration_in_traffic_text': route.get('duration_in_traffic', {}).get('text', None),
                    'traffic_ratio': self._calculate_traffic_ratio(route),
                    'speed_kmh': self._calculate_speed(route),
                    'congestion_level': self._classify_congestion(route)
                }
            else:
                print(f"API Error: {data['status']}")
                return None

        except Exception as e:
            print(f"Error fetching traffic data: {e}")
            return None

    def _calculate_traffic_ratio(self, route):
        """Calculate traffic congestion ratio"""
        if 'duration_in_traffic' in route:
            normal_duration = route['duration']['value']
            traffic_duration = route['duration_in_traffic']['value']
            return traffic_duration / normal_duration if normal_duration > 0 else 1.0
        return 1.0

    def _calculate_speed(self, route):
        """Calculate average speed in km/h"""
        distance_km = route['distance']['value'] / 1000

        if 'duration_in_traffic' in route:
            duration_hours = route['duration_in_traffic']['value'] / 3600
        else:
            duration_hours = route['duration']['value'] / 3600

        return distance_km / duration_hours if duration_hours > 0 else 0

    def _classify_congestion(self, route):
        """Classify congestion level based on traffic ratio"""
        traffic_ratio = self._calculate_traffic_ratio(route)

        if traffic_ratio < 1.1:
            return 'light'
        elif traffic_ratio < 1.3:
            return 'moderate'
        elif traffic_ratio < 1.5:
            return 'heavy'
        else:
            return 'severe'

    def collect_historical_data(self, origin, destination, days=7, interval_hours=1):
        """
        Collect traffic data for the current and future times only.
        Note: The Google Maps API does not support querying for past times.
        To collect real historical data, you must run this function periodically (e.g., with a scheduler) and store the results over time.

        Args:
            origin (str): Starting location
            destination (str): Ending location
            days (int): Number of days to collect data (only used for scheduling, not for past data)
            interval_hours (int): Hours between data collection

        Returns:
            list: List of traffic data records
        """
        data_records = []

        # Only collect for now and future times
        current_time = datetime.now()
        end_time = current_time + timedelta(days=days)

        print(f"Collecting traffic data from {current_time} to {end_time}")

        while current_time <= end_time:
            print(f"Collecting data for: {current_time}")

            traffic_data = self.get_traffic_data(origin, destination, current_time)

            if traffic_data:
                traffic_data.update({
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'month': current_time.month,
                    'is_weekend': 1 if current_time.weekday() >= 5 else 0,
                    'is_rush_hour': 1 if current_time.hour in [7, 8, 9, 17, 18, 19] else 0
                })
                data_records.append(traffic_data)

            # Move to next time point
            current_time += timedelta(hours=interval_hours)

            # Add delay to avoid hitting API rate limits
            time.sleep(0.1)

        return data_records

class SimpleTrafficPredictor:
    """
    Simple traffic prediction model using Random Forest
    """
    def __init__(self):
        # Tune Random Forest: more trees, limit max_depth, use min_samples_leaf
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []

    def prepare_features(self, data):
        """
        Prepare features for machine learning

        Args:
            data (list): List of traffic data records

        Returns:
            pandas.DataFrame: Processed features
        """
        df = pd.DataFrame(data)

        # Data cleaning: Remove outliers and missing values
        df = df.dropna(subset=['duration_in_traffic_seconds'])
        # Remove negative or zero durations/distances
        df = df[(df['duration_in_traffic_seconds'] > 0) & (df['distance_meters'] > 0)]
        # Remove extreme outliers (top/bottom 1%)
        for col in ['duration_in_traffic_seconds', 'distance_meters', 'speed_kmh']:
            if col in df.columns:
                q_low = df[col].quantile(0.01)
                q_high = df[col].quantile(0.99)
                df = df[(df[col] >= q_low) & (df[col] <= q_high)]

        # Select features
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 
                       'distance_meters', 'duration_seconds']

        # Add more lag features (previous 1, 2, 3 hours if available)
        df = df.sort_values('timestamp').reset_index(drop=True)
        for col in ['traffic_ratio', 'speed_kmh']:
            for lag in [1, 2, 3]:
                lag_col = f'{col}_lag{lag}'
                df[lag_col] = df[col].shift(lag)
                feature_cols.append(lag_col)

        # Remove rows with NaN values (from lagging)
        df = df.dropna()

        self.feature_columns = feature_cols
        return df[feature_cols], df

    def train(self, historical_data, target_column='duration_in_traffic_seconds'):
        """
        Train the prediction model

        Args:
            historical_data (list): Historical traffic data
            target_column (str): Column to predict
        """
        print("Preparing features...")
        X, df = self.prepare_features(historical_data)
        y = df[target_column]

        if len(X) < 10:
            print("Not enough data for training (minimum 10 samples required)")
            return None

        print(f"Training with {len(X)} samples and {len(X.columns)} features")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        print(f"Training MAE: {train_mae:.2f} seconds")
        print(f"Testing MAE: {test_mae:.2f} seconds")
        print(f"Testing R²: {test_r2:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())

        self.is_trained = True
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }

    def predict(self, current_conditions):
        """
        Predict future traffic conditions

        Args:
            current_conditions (dict): Current traffic and temporal conditions

        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        features = []
        for col in self.feature_columns:
            if col in current_conditions:
                features.append(current_conditions[col])
            else:
                features.append(0)  # Default value for missing features

        # Scale and predict
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]

        return {
            'predicted_duration_seconds': prediction,
            'predicted_duration_minutes': prediction / 60,
            'confidence': 'medium'  # Simplified confidence measure
        }


# OOP: Encapsulate the workflow in a TrafficPredictionSystem class
class TrafficPredictionSystem:
    def collect_real_data(self, days=1, interval_hours=1):
        print("\nCollecting real traffic data from Google Maps API...")
        real_data = self.collector.collect_historical_data(self.origin, self.destination, days=days, interval_hours=interval_hours)
        return real_data
    def __init__(self, api_key, origin, destination):
        self.api_key = api_key
        self.origin = origin
        self.destination = destination
        self.collector = GoogleMapsTrafficCollector(api_key)
        self.predictor = SimpleTrafficPredictor()
        self.current_data = None
        self.model_results = None

    def get_current_traffic(self):
        self.current_data = self.collector.get_traffic_data(self.origin, self.destination)
        return self.current_data

    def create_synthetic_historical_data(self, days=7):
        synthetic_data = []
        base_time = datetime.now() - timedelta(days=days)
        for i in range(days * 24):
            time_point = base_time + timedelta(hours=i)
            hour = time_point.hour
            day_of_week = time_point.weekday()
            base_duration = 600
            if hour in [7, 8, 9, 17, 18, 19]:
                traffic_multiplier = 1.5 + np.random.normal(0, 0.2)
            elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                traffic_multiplier = 0.8 + np.random.normal(0, 0.1)
            else:
                traffic_multiplier = 1.0 + np.random.normal(0, 0.15)
            if day_of_week >= 5:
                traffic_multiplier *= 0.9
            duration_in_traffic = base_duration * traffic_multiplier
            synthetic_record = {
                'timestamp': time_point,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': time_point.month,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'is_rush_hour': 1 if hour in [7, 8, 9, 17, 18, 19] else 0,
                'distance_meters': 2000,
                'duration_seconds': base_duration,
                'duration_in_traffic_seconds': duration_in_traffic,
                'traffic_ratio': duration_in_traffic / base_duration,
                'speed_kmh': 7.2 / (duration_in_traffic / 3600),
                'congestion_level': 'moderate'
            }
            synthetic_data.append(synthetic_record)
        for i in range(1, len(synthetic_data)):
            synthetic_data[i]['traffic_ratio_lag1'] = synthetic_data[i-1]['traffic_ratio']
            synthetic_data[i]['speed_kmh_lag1'] = synthetic_data[i-1]['speed_kmh']
        return synthetic_data

    def train_model(self, historical_data):
        self.model_results = self.predictor.train(historical_data[1:])
        return self.model_results

    def predict_future(self, hours_ahead=6):
        if not self.current_data or not self.model_results:
            print("Model or current data not available.")
            return
        print("\n=== Traffic Prediction Output ===")
        print(f"Model Accuracy (R²): {self.model_results['test_r2']:.3f}")
        print(f"Test MAE: {self.model_results['test_mae']:.1f} seconds")
        print("\nPredicted Traffic Conditions for the Next {} Hours:".format(hours_ahead))
        verdicts = []
        for i in range(1, hours_ahead+1):
            future_time = datetime.now() + timedelta(hours=i)
            prediction_conditions = {
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'month': future_time.month,
                'is_weekend': 1 if future_time.weekday() >= 5 else 0,
                'is_rush_hour': 1 if future_time.hour in [7, 8, 9, 17, 18, 19] else 0,
                'distance_meters': self.current_data['distance_meters'],
                'duration_seconds': self.current_data['duration_seconds'],
                'traffic_ratio_lag1': self.current_data['traffic_ratio'],
                'speed_kmh_lag1': self.current_data['speed_kmh']
            }
            prediction = self.predictor.predict(prediction_conditions)
            # Determine traffic level verdict
            normal_duration = self.current_data['duration_seconds']
            predicted = prediction['predicted_duration_seconds']
            ratio = predicted / normal_duration if normal_duration > 0 else 1.0
            if ratio < 1.1:
                verdict = 'light traffic'
            elif ratio < 1.3:
                verdict = 'moderate traffic'
            elif ratio < 1.5:
                verdict = 'heavy traffic'
            else:
                verdict = 'severe traffic'
            verdicts.append(verdict)
            print(f"  At {future_time.strftime('%Y-%m-%d %H:%M')}: {prediction['predicted_duration_minutes']:.1f} min (Accuracy: {self.model_results['test_r2']:.2f} R²)")
        print("\n=== Final Traffic Verdict for Each Hour ===")
        for i, verdict in enumerate(verdicts, 1):
            print(f"{i} hour: {verdict}")
        print("\n=== End of Prediction ===\n")

def main():
    print("=== Traffic Prediction Prototype Demo ===\n")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    if not API_KEY or API_KEY.strip() == "":
        print("⚠️  Please set your Google Maps API key in the .env file!")
        print("   Get one at: https://developers.google.com/maps/documentation/directions/get-api-key")
        return
    # Take user input for origin and destination
    origin = input("Enter the start location (origin): ").strip()
    destination = input("Enter the end location (destination): ").strip()
    if not origin or not destination:
        print("Origin and destination cannot be empty.")
        return
    system = TrafficPredictionSystem(API_KEY, origin, destination)
    print(f"Route: {origin} → {destination}\n")
    print("1. Getting current traffic data...")
    current_data = system.get_current_traffic()
    if current_data:
        print(f"   Distance: {current_data['distance_text']}")
        print(f"   Normal Duration: {current_data['duration_text']}")
        print(f"   With Traffic: {current_data['duration_in_traffic_text']}")
        print(f"   Speed: {current_data['speed_kmh']:.1f} km/h")
        print(f"   Congestion: {current_data['congestion_level']}")
        print(f"   Traffic Ratio: {current_data['traffic_ratio']:.2f}")
    else:
        print("   Failed to get current traffic data")
        return
    print("\n2. Creating synthetic historical data for demo...")
    print("\nWould you like to use real Google Maps API data or synthetic data for training?")
    print("1. Real API data (slower, uses your API quota)")
    print("2. Synthetic data (fast demo)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        days = input("How many days of real data to collect? (default 1): ").strip()
        interval = input("Interval in hours between samples? (default 1): ").strip()
        days = int(days) if days.isdigit() and int(days) > 0 else 1
        interval = int(interval) if interval.isdigit() and int(interval) > 0 else 1
        real_data = system.collect_real_data(days=days, interval_hours=interval)
        print(f"   Collected {len(real_data)} real records")
        data_for_training = real_data
    else:
        synthetic_data = system.create_synthetic_historical_data(days=7)
        print(f"   Created {len(synthetic_data)} synthetic records")
        data_for_training = synthetic_data
    print("\n3. Training prediction model...")
    results = system.train_model(data_for_training)
    if results:
        system.predict_future(hours_ahead=6)


if __name__ == "__main__":
    main()
