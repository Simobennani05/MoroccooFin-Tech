import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import io
import base64
import os

# Fonction pour dédupliquer les colonnes
def dedup_columns(cols):
    seen = {}
    result = []
    for col in cols:
        col_clean = col.strip()
        if col_clean not in seen:
            seen[col_clean] = 0
            result.append(col_clean)
        else:
            seen[col_clean] += 1
            result.append(f"{col_clean}_{seen[col_clean]}")
    return result

# Fonction pour traiter les fichiers uploadés
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'xlsx' in filename.lower() or 'xls' in filename.lower():
        return pd.read_excel(io.BytesIO(decoded))
    return None

# Initialiser l'application Dash
app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'])

app.layout = html.Div(className='container mx-auto p-4', children=[
    html.H1('Prédiction des Performances OPCVM', className='text-3xl font-bold mb-4 text-center'),
    html.Div([
        html.Label('Fichier performances OPCVM:', className='text-lg font-semibold'),
        dcc.Upload(
            id='upload-opcvm',
            multiple=False,
            children=html.Button('Sélectionner un fichier', className='p-2 bg-blue-500 text-white rounded'),
            className='mb-4'
        ),
    ]),
    html.Div([
        html.Label('Fichier indices pour prédiction:', className='text-lg font-semibold'),
        dcc.Upload(
            id='upload-unseen',
            multiple=False,
            children=html.Button('Sélectionner un fichier', className='p-2 bg-blue-500 text-white rounded'),
            className='mb-4'
        ),
    ]),
    html.Label('Modèle de prédiction:', className='text-lg font-semibold'),
    dcc.Dropdown(
        id='model-choice',
        options=[
            {'label': 'XGBoost', 'value': 'xgb'},
            {'label': 'Random Forest', 'value': 'rf'},
            {'label': 'Linear Regression', 'value': 'lr'},
            {'label': 'LightGBM', 'value': 'lgbm'},
            {'label': 'CatBoost', 'value': 'catboost'}
        ],
        value='xgb',
        className='mb-4 p-2 border rounded'
    ),
    html.Label('Entrer un OPCVM:', className='text-lg font-semibold'),
    dcc.Input(
        id='opcvm-input',
        type='text',
        placeholder='Ex: UPLINE ACTIONS',
        className='mb-4 p-2 border rounded'
    ),
    dcc.Graph(id='prediction-plot'),
    dcc.Graph(id='feature-importance'),
    html.Button("Télécharger les prédictions", id="download-button", className="p-2 bg-green-500 text-white rounded my-2"),
    dcc.Download(id="download-data"),
    html.Label('Résultats Textuels:', className='text-lg font-semibold mt-4'),
    dcc.Textarea(
        id='results-text',
        value='',
        style={'width': '100%', 'height': '200px', 'resize': 'vertical'},
        className='p-2 border rounded'
    )
])

@app.callback(
    [Output('prediction-plot', 'figure'),
     Output('feature-importance', 'figure'),
     Output('results-text', 'value'),
     Output('download-data', 'data')],
    [Input('opcvm-input', 'value'),
     Input('upload-opcvm', 'contents'),
     Input('upload-unseen', 'contents'),
     Input('model-choice', 'value'),
     Input('download-button', 'n_clicks')],
    [State('upload-opcvm', 'filename'),
     State('upload-unseen', 'filename')]
)
def predict_dashboard(opcvm_input, opcvm_contents, unseen_contents, model_choice, download_clicks, opcvm_filename, unseen_filename):
    default_figure = go.Figure()
    if not opcvm_input:
        return default_figure, default_figure, 'Veuillez entrer un OPCVM valide.', None

    df_opcvm = parse_contents(opcvm_contents, opcvm_filename) if opcvm_contents else None
    df_unseen = parse_contents(unseen_contents, unseen_filename) if unseen_contents else None
    if df_opcvm is None or df_unseen is None:
        return default_figure, default_figure, 'Veuillez télécharger les deux fichiers.', None

    df_opcvm.columns = dedup_columns(df_opcvm.columns)
    date_col = next((col for col in df_opcvm.columns if 'date' in col.lower()), None)
    vl_col = next((col for col in df_opcvm.columns if col.strip().upper() == 'VL'), None)
    opcvm_col = next((col for col in df_opcvm.columns if 'opcvm' in col.lower()), None)
    df_opcvm = df_opcvm.rename(columns={date_col: 'Date', vl_col: 'VL', opcvm_col: 'OPCVM'})
    df_opcvm['VL'] = df_opcvm['VL'].astype(str).str.replace(',', '.').str.replace(' ', '').str.replace('%', '').str.strip().apply(pd.to_numeric, errors='coerce')
    df_opcvm['Date'] = pd.to_datetime(df_opcvm['Date'], errors='coerce')
    df_opcvm = df_opcvm.sort_values(['OPCVM', 'Date'])
    df_opcvm['perf_fonds'] = df_opcvm.groupby('OPCVM')['VL'].transform(lambda x: x.pct_change(fill_method=None) * 100).ffill()

    if opcvm_input not in df_opcvm['OPCVM'].values:
        return default_figure, default_figure, 'OPCVM non trouvé.', None

    df_unseen = df_unseen.rename(columns={'DATE_COTATION': 'Date', 'VALEUR_PUBLIEE': 'Valeur'})
    df_unseen['Date'] = pd.to_datetime(df_unseen['Date'], errors='coerce')
    # Pivot des indices
    df_unseen_pivot = (
        df_unseen.pivot(index="Date", columns="INDICE", values="Valeur")
        .sort_index()
    )

    # Calcul des rendements en %
    df_unseen_perf = df_unseen_pivot.pct_change(fill_method=None) * 100

    # On garde uniquement les indices souhaités
    selected_indices = ["MBICT", "MBIMT", "MBIMTLT", "MBILT", "MASIRB"]
    df_unseen_perf = df_unseen_perf[selected_indices]

    # Définition des features utilisées par le modèle
    features = selected_indices

    df_unseen_perf = df_unseen_pivot.pct_change(fill_method=None) * 100

    df_fund = df_opcvm[df_opcvm['OPCVM'] == opcvm_input].copy()
    df_fund['Date'] = pd.to_datetime(df_fund['Date'])
    df_fund = df_fund.sort_values('Date')

    df_perf_reset = df_unseen_perf.reset_index()
    features = list(df_unseen_perf.columns)
    df_merged = pd.merge_asof(df_fund, df_perf_reset, on='Date', direction='nearest', tolerance=pd.Timedelta(days=1))
    df_merged.dropna(subset=['perf_fonds'], inplace=True)
    df_merged[features] = df_merged[features].ffill()

    if df_merged.empty:
        return default_figure, default_figure, 'Pas assez de données pour entraîner le modèle.', None

    X = df_merged[features]
    y = df_merged['perf_fonds']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = None
    if model_choice == 'xgb':
        model = GridSearchCV(XGBRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [3, 5]}, cv=3)
    elif model_choice == 'rf':
        model = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [100, 200]}, cv=3)
    elif model_choice == 'lr':
        model = LinearRegression()
    elif model_choice == 'lgbm':
        model = LGBMRegressor()
    elif model_choice == 'catboost':
        model = CatBoostRegressor(verbose=0)
    else:
        return default_figure, default_figure, 'Modèle non reconnu.', None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    X_future = scaler.transform(df_unseen_perf.fillna(0))
    predictions = model.predict(X_future)

    df_export = pd.DataFrame({'Date': df_unseen_perf.index, 'Prediction': predictions})

    prediction_fig = px.line(df_export, x='Date', y='Prediction', title=f'Prédictions pour {opcvm_input}')
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_

    if hasattr(model, 'feature_importances_'):
        importance_fig = px.bar(x=features, y=model.feature_importances_, labels={'x': 'Indice', 'y': 'Importance'}, title='Importance des Indices')
    elif hasattr(model, 'coef_'):
        importance_fig = px.bar(x=features, y=model.coef_, labels={'x': 'Indice', 'y': 'Coefficient'}, title='Poids des Indices')
    else:
        importance_fig = default_figure

    results_text = f"""
    ### Résultats pour {opcvm_input}
    - **Modèle** : {model_choice.upper()}
    - **Période de prédiction** : {df_unseen_perf.index.min().date()} → {df_unseen_perf.index.max().date()}
    - **Nombre de jours prédits** : {len(predictions)}
    - **MSE (test)** : {mse:.4f}
    - **R² Score (test)** : {r2:.4f}
    """

    excel_data = df_export.to_excel(index=False, engine='openpyxl')
    return prediction_fig, importance_fig, results_text, dcc.send_data_frame(df_export.to_excel, filename="predictions.xlsx", index=False)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
