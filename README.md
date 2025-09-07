# 🚦 Traffic Predictor  

Predict traffic conditions across different locations using **historical data** and **machine learning models**.  
This project focuses on **time-series traffic prediction**, leveraging temporal, spatial, and lag-based features.  

---

## 📑 Table of Contents  

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Modeling Approach](#modeling-approach)  
- [Results](#results)  
- [Future Work](#future-work)  
- [License](#license)  

---

## 📖 Overview  

Traffic congestion is a growing issue in urban areas—causing **delays, fuel waste, and pollution**.  
This project aims to predict traffic conditions in advance, helping **drivers, city planners, and navigation systems** make informed decisions.  

The pipeline uses **historical traffic data** to predict future traffic levels at given locations and timestamps, applying **machine learning** and **time-series forecasting techniques** to capture both **temporal patterns** and **location-specific trends**.  

---

## 📊 Dataset  

The dataset contains multi-location traffic records with timestamps.  

**Key columns include:**  
- **Location Code** → Encoded ID for locations (e.g., Manhattan = 1, Bronx = 2)  
- **Timestamp** → Date and time of traffic measurement  
- **Traffic Lag Features** → Historical values used as predictors  
- **Traffic Metric (Target)** → Traffic level / speed / congestion  

📂 Dataset Source: https://kaggle.com/datasets/690c165527ba94ba77c5f026357264141ee3660e8cebe0581f9c7cfb2e05770e

---

## ✨ Features  

- ✅ Location-based traffic encoding  
- ✅ Temporal features (hour, weekday, weekend, seasonality)  
- ✅ Lag features for historical traffic readings  
- ✅ Optional external factors (weather, holidays, events)  

---

## ⚙️ Installation  

Clone the repository and install dependencies:  

```bash
git clone <repository_link>
cd traffic-predictor
pip install -r requirements.txt
🚀 Usage
Run preprocessing, training, prediction, and evaluation scripts:

bash
Copy code
python preprocess.py
python train.py
python predict.py
python evaluate.py
🧠 Modeling Approach
Data Preprocessing

Handle missing values

Encode categorical features

Create lag-based features

Train-Test Split

Chronological split (80% training, 20% testing)

Models Tested

Random Forest

XGBoost

LSTM (Deep Learning)

Evaluation Metrics

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Accuracy (for classification-style predictions)
```

## 📈 Results  

The model’s performance was evaluated using **precision, recall, and F1-score** across three traffic categories.  

| Class     | Precision | Recall | F1-Score | Support   |
|-----------|-----------|--------|----------|-----------|
| Heavy     | 0.90      | 0.87   | 0.89     | 1,261,297 |
| Light     | 0.09      | 0.44   | 0.15     | 6,574     |
| Moderate  | 0.76      | 0.79   | 0.77     | 654,680   |

**Overall Metrics**  

| Metric        | Score |
|---------------|-------|
| Accuracy      | 0.84  |
| Macro Avg F1  | 0.60  |
| Weighted Avg F1 | 0.85 |

---

📌 **Insights:**  
- The model performs strongly on **Heavy** and **Moderate** traffic conditions.  
- Performance on **Light** traffic is weaker, likely due to class imbalance (very few samples).  
- Overall accuracy is **84%**, with a weighted F1 of **0.85**, showing robustness on larger traffic classes.

🔮 Future Work

- Add weather & event data for richer predictions
- Deploy a real-time dashboard for traffic visualization
- Experiment with advanced architectures (e.g., Transformers for time-series forecasting)
  
📜 License

This project is licensed under the MIT License.

Contributing 🤝
Encourage developers to contribute.

## 🤝 Contributing  

Contributions are welcome! 🎉  
If you’d like to improve this project, please fork the repo and create a pull request.  
For major changes, open an issue first to discuss what you’d like to change.  


Acknowledgments / Credits 🙌
Thank datasets, libraries, or people who helped.

## 🙌 Acknowledgments  

- Inspired by real-world traffic prediction research  
- Thanks to the open-source community for tools like **scikit-learn**, **XGBoost**, and **TensorFlow**  


Contact 📬
Let users know how to reach you.

## 📬 Contact  

Created by [Deva Nandan](https://github.com/Xer0oo7), [Aditya Shankar](https://github.com/photon457) – feel free to connect! 
