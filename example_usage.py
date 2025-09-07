
"""
Example Usage of Traffic Prediction Prototype

This script demonstrates how to use the traffic prediction system
with your own routes and data collection schedule.
"""

# Ensure local imports work regardless of run location
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv


# Import the OOP system class
from traffic_prediction_prototype import TrafficPredictionSystem

def collect_and_predict_traffic():
    """
    Example function showing how to collect data and make predictions
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')

    if not api_key:
        print("Please set GOOGLE_MAPS_API_KEY in your .env file")
        return


    # Define your route (replace with your real addresses if desired)
    origin = "Times Square, New York, NY"
    destination = "Central Park, New York, NY"

    print(f"Monitoring route: {origin} → {destination}")

    # Initialize traffic prediction system
    system = TrafficPredictionSystem(api_key, origin, destination)

    # Get current traffic conditions
    print("\n1. Current Traffic Conditions:")
    current_traffic = system.get_current_traffic()

    if current_traffic:
        print(f"   Distance: {current_traffic['distance_text']}")
        print(f"   Duration (no traffic): {current_traffic['duration_text']}")
        print(f"   Duration (with traffic): {current_traffic['duration_in_traffic_text']}")
        print(f"   Congestion Level: {current_traffic['congestion_level']}")
        print(f"   Average Speed: {current_traffic['speed_kmh']:.1f} km/h")

    # Collect historical data (you would run this over multiple days)
    print("\n2. Collecting Historical Data...")
    # For demo, use synthetic data for OOP consistency
    historical_data = system.create_synthetic_historical_data(days=2)
    print(f"   Created {len(historical_data)} synthetic records")

    if len(historical_data) > 10:
        print(f"   Collected {len(historical_data)} data points")

        # Train prediction model
        print("\n3. Training Prediction Model...")
        results = system.train_model(historical_data)

        if results:
            # Make predictions
            system.predict_future(hours_ahead=3)
    else:
        print("   Not enough data for training. Need at least 10 data points.")

if __name__ == "__main__":
    collect_and_predict_traffic()
