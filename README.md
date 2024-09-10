Predicting Product Changes for Bank Clients: A Hybrid Approach with LightGBM and TSMixer
Overview
This project aims to predict product changes for bank clients using a combination of machine learning models, including LightGBM and TSMixer. The approach blends tree-based models with neural architectures to handle time-series data more effectively, providing a robust solution for predicting customer behavior in financial services.

Features
Data Preprocessing: Handles missing values, categorical encoding, and feature engineering.
LightGBM Model: Hyperparameter optimization using Optuna for efficient tuning and improved performance on product change predictions.
TSMixer: Time-series model for capturing complex patterns and forecasting product changes over a horizon of 96 steps.
Evaluation: Performance metrics including Precision, Recall, F1 Score, MAE, and MSE to assess model effectiveness across multiple target labels.
Installation
bash
Copy code
pip install pandas numpy seaborn matplotlib xgboost shap-hypetune optuna lightgbm ray neuralforecast datasetsforecast statsmodels
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/bank-client-product-change-prediction.git
cd bank-client-product-change-prediction
Run the notebook in your preferred environment (e.g., Jupyter, Google Colab).
Key Findings
LightGBM performs well on structured data but struggles with sparse features.
TSMixer outperforms traditional methods in time-series forecasting and complements LightGBM when used in hybrid or ensembling strategies.
Hybrid Approach: Combining LightGBM with TSMixer yields more consistent predictions, especially for imbalanced and time-sensitive datasets.
Future Improvements
Exploring additional blending, ensembling, or stacking techniques with other models like LSTM could enhance performance further, especially on sparse or imbalanced datasets.
