MoroccoFin-Tech
Project Description
MoroccoFin-Tech is a collection of Python-based tools developed during a three-month internship (June–August 2025) at Upline Capital Management (UCM), a leading Moroccan asset management firm affiliated with Groupe Banque Populaire. These tools leverage machine learning and financial modeling to enhance competitive intelligence for Organismes de Placement Collectif en Valeurs Mobilières (OPCVM) and streamline bond valuation, supporting data-driven investment decisions.
The repository includes three key components:

Maturity/Titles Dashboard (maturite.py): An interactive Streamlit dashboard predicting daily OPCVM returns based on maturities and titles. Using Linear Regression, Random Forest, and Gradient Boosting, it achieves an R² score of 0.82, enabling portfolio composition analysis through file uploads and visualizations.
Multi-Model Dashboard (multimodel_dashboard.py): A Dash-based tool forecasting OPCVM performance using Moroccan bond indices (MBICT, MBIMT, MBIMTLT, MBILT). It compares XGBoost, LightGBM, CatBoost, Random Forest, and Linear Regression, with XGBoost yielding an R² of 0.87. Features include interactive Plotly charts and Excel exports.
Bond Pricer (pricer.py): A Streamlit application valuing bonds via an actuarial yield curve. It supports single or batch processing, calculates discounted cash flows, and visualizes results. For example, a 7-year bond with a 100,000 MAD nominal and 5% coupon is valued at 102,345 MAD.

Built with Python, pandas, scikit-learn, Dash, Streamlit, and Plotly, these tools address challenges like data quality and performance optimization, replacing manual processes at UCM. The repository requires proprietary financial data (not included) and is licensed under MIT. Contributions are welcome via pull requests. For setup and usage details, see below.
Word count: 350
Prerequisites

Python: Version 3.8 or higher
Dependencies: Listed in requirements.txt. Key libraries include:
pandas for data manipulation
scikit-learn, xgboost, lightgbm, catboost for machine learning
dash, streamlit, plotly for interactive visualizations
scipy for interpolation in bond pricing
chardet for encoding detection


Data Files: Excel/CSV files containing OPCVM performance, maturity/title data, MBI indices, and yield curves (not included due to proprietary nature).

Installation

Clone the Repository:
git clone https://github.com/[Your-GitHub-Username]/MoroccoFin-Tech.git
cd MoroccoFin-Tech


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Prepare Data:

For maturite.py: Provide Excel/CSV files (e.g., matrice.xlsx, merged_performance_daily.xlsx) with OPCVM and maturity/title data.
For multimodel_dashboard.py: Provide Excel files with OPCVM performance (VL column) and MBI indices.
For pricer.py: Provide a CSV file for the yield curve (Echeance, Taux) and an Excel file for bond details (NOMINAL, VALEUR_TAUX, DATE_ECHEANCE).



Usage
1. Maturity/Titles Dashboard (maturite.py)

Purpose: Predict daily OPCVM returns based on maturities and titles.
Run:streamlit run maturite.py


Features:
Upload Excel/CSV files via Streamlit interface.
Select an OPCVM and ML model (Linear Regression, Random Forest, Gradient Boosting).
View predicted vs. actual returns and asset weights.
Export results to CSV.


Example Output: R² score of 0.82 (Random Forest) for a daily return prediction of 0.5% ± 0.03%.

2. Multi-Model Dashboard (multimodel_dashboard.py)

Purpose: Forecast OPCVM performance using MBI indices and compare ML models.
Run:python multimodel_dashboard.py


Features:
Upload Excel files with OPCVM and MBI data.
Select an OPCVM and ML model (XGBoost, LightGBM, CatBoost, Random Forest, Linear Regression).
Visualize predictions and feature importances with Plotly charts.
Export predictions to Excel.


Example Output: R² score of 0.87 (XGBoost) for a 30-day return prediction of 0.7%.

3. Bond Pricer (pricer.py)

Purpose: Value bonds using an actuarial yield curve.
Run:streamlit run pricer.py


Features:
Input bond parameters (nominal, coupon rate, maturity, frequency) or upload an Excel file for batch processing.
Upload a yield curve CSV.
Visualize cash flows and yield curve.
Export valuations to Excel.


Example Output: A 7-year bond with 100,000 MAD nominal and 5% coupon valued at 102,345 MAD.

File Structure
MoroccoFin-Tech/
├── maturite.py              # Dashboard for predicting OPCVM returns based on maturities/titles
├── multimodel_dashboard.py   # Dashboard for forecasting OPCVM performance using MBI indices
├── pricer.py                # Bond valuation tool using actuarial yield curves
├── requirements.txt         # Python dependencies
├── README.md                # This file

Dependencies
Install the required packages using:
pip install pandas scikit-learn xgboost lightgbm catboost dash streamlit plotly scipy chardet

Alternatively, use the provided requirements.txt:
pip install -r requirements.txt

Data Requirements

Maturity/Titles Dashboard: Excel/CSV files with columns for date, opcvm, 1jour (daily return), and maturity/title performance (e.g., short/medium/long-term yields).
Multi-Model Dashboard: Excel files with OPCVM performance (VL column) and MBI indices (Date, INDICE, value columns).
Bond Pricer: CSV for yield curve (Echeance, Taux) and Excel for bonds (NOMINAL, VALEUR_TAUX, DATE_ECHEANCE, etc.).
Note: Sample data is not included due to proprietary restrictions. Users must provide their own financial datasets.

Technical Details

Maturity/Titles Dashboard:
Preprocessing: Merges data on date, removes NaN with dropna, deduplicates columns.
Models: Linear Regression, Random Forest, Gradient Boosting (80/20 train-test split, random_state=42).
Interface: Streamlit with file upload, model selection, and result visualization.


Multi-Model Dashboard:
Preprocessing: Converts VL to returns (pct_change * 100), normalizes with StandardScaler, merges with pd.merge_asof.
Models: XGBoost, LightGBM, CatBoost, Random Forest, Linear Regression (optimized with GridSearchCV).
Interface: Dash with Tailwind CSS, Plotly visualizations, and Excel export.


Bond Pricer:
Preprocessing: Handles encoding with chardet, converts rates and dates.
Logic: Uses scipy.interpolate.interp1d for yield curve interpolation, calculates discounted cash flows.
Interface: Streamlit with manual input, batch processing, and Plotly charts.



Challenges Addressed

Data Quality: Handled missing values and inconsistent formats (e.g., date formats) with interpolation and validation.
Performance: Optimized large matrix computations with batch processing and ffill.
Compatibility: Resolved encoding issues with chardet and Python package conflicts using virtual environments.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make changes and commit (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a pull request with a detailed description of your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, open an issue on GitHub.
