# Predicting Product Changes for Bank Clients  
**A Hybrid Approach with LightGBM and TSMixer**

## Overview  
This project tackles the challenging task of predicting product changes for bank clients by combining tree-based models (LightGBM) with time-series neural architectures (TSMixer). By integrating these approaches, the model can better capture both structured and temporal aspects of customer behavior, offering a robust solution in the financial services domain.  

![image](https://github.com/user-attachments/assets/8906a839-e60b-430b-853a-31c2d14f957f)


## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Notebook & Code Overview](#notebook--code-overview)  
- [Key Findings](#key-findings)  
- [Future Improvements](#future-improvements)  
- [License](#license)  
- [Contact](#contact)

## Features  
- **Data Preprocessing:**  
  Handles missing values, categorical encoding, and feature engineering tailored for time-series data.  

- **LightGBM Model:**  
  Utilizes hyperparameter optimization via Optuna, improving performance on product change predictions.  

- **TSMixer & TSMixerx:**  
  Neural network architectures that capture complex temporal dependencies across a forecasting horizon of 96 time steps.  

- **Evaluation Metrics:**  
  Uses Precision, Recall, F1 Score, MAE, and MSE to assess model performance across multiple target labels.

![image](https://github.com/user-attachments/assets/5a4549a7-ad72-4280-858c-9348da63b6c7)


## Installation  
Ensure you have Python 3.7+ installed. Then, install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib xgboost shap-hypetune optuna lightgbm ray neuralforecast datasetsforecast statsmodels
```

*For further details on environment setup, see guidelines from similar data science projects [citeturn0search7].*

## Usage  
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/yourusername/bank-client-product-change-prediction.git
   cd bank-client-product-change-prediction
   ```
2. **Run the Notebook:**  
   Open the Jupyter Notebook (or your preferred environment) to explore the complete workflow—from data preprocessing to model evaluation.
3. **Experiment and Modify:**  
   The repository includes comprehensive code cells for exploratory data analysis (EDA), feature engineering, model training, and evaluation. Feel free to adjust the code to suit your needs.

## Notebook & Code Overview  
The repository contains a detailed Jupyter Notebook that walks you through:  
- **Data Cleaning & Preprocessing:**  
  Including renaming columns, handling missing values, and creating time-based features (e.g., weekly, quarterly lags).  
- **Exploratory Data Analysis (EDA):**  
  Visualizations such as autocorrelation plots, product ownership trends, and heatmaps to uncover underlying data patterns.  
- **Feature Engineering & Selection:**  
  Techniques such as lag features and RFE (Recursive Feature Elimination) with XGBoost to identify significant predictors.  
- **Modeling:**  
  Implementation of LightGBM with hyperparameter tuning via Optuna, and time-series forecasting using TSMixer variants and MLPMultivariate models.  
- **Evaluation:**  
  Detailed performance evaluation using standard metrics along with visualizations of loss functions and confusion matrices.

*For structuring your code and documentation, refer to examples from top GitHub data science projects [citeturn0search8].*

## Key Findings  
- **Hybrid Approach Advantage:**  
  Combining LightGBM with TSMixer yields consistent predictions—especially beneficial when dealing with imbalanced or sparse datasets.  
- **LightGBM Performance:**  
  Excels in structured data predictions but shows challenges with highly sparse features.  
- **TSMixer Models:**  
  Deliver lower MAE and MSE over a forecasting horizon of 96 steps, highlighting superior handling of temporal patterns.

## Future Improvements  
- **Model Ensembling:**  
  Explore stacking techniques that integrate LightGBM with additional models (e.g., LSTM) to further address sparse data challenges.  
- **Enhanced Feature Engineering:**  
  Incorporate additional temporal features (e.g., moving averages, Fourier transformations) to capture seasonality and cyclical trends.  
- **Extended Hyperparameter Tuning:**  
  Leverage advanced optimization frameworks to further refine model performance.

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact  
For questions, feedback, or collaboration opportunities, please contact:  

**Shoval Benjer**  
Creative Data Scientist | Tel Aviv - Jaffa, ISR  
GitHub: [ShovalBenjer](https://github.com/ShovalBenjer)  
Email: shovalb9@gmail.com  
