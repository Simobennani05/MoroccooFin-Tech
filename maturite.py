import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Titre du dashboard
st.title("Dashboard Interactif pour Prédiction de Composition des OPCVM")

# Description brève
st.write("""
Ce dashboard permet de téléverser deux fichiers : un pour les performances journalières des maturités/titres (matrice.xlsx ou similaire) 
et un pour les performances des OPCVM (merged_performance_daily.xlsx ou similaire).
Il entraîne un modèle ML pour prédire les rendements des OPCVM et estimer leur composition (via coefficients ou importances).
""")

# Téléversement des deux fichiers
uploaded_assets = st.file_uploader("Téléversez le fichier des maturités/titres (Excel ou CSV)", type=["csv", "xlsx"])
uploaded_opcvm = st.file_uploader("Téléversez le fichier des OPCVM (Excel ou CSV)", type=["csv", "xlsx"])

if uploaded_assets is not None and uploaded_opcvm is not None:
    # Étape 1: Lecture du fichier assets (maturités/titres)
    if uploaded_assets.name.endswith('.csv'):
        asset_df = pd.read_csv(uploaded_assets)
    else:
        # Pour Excel, lecture de toutes les sheets et concat si multiple
        asset_sheets = pd.read_excel(uploaded_assets, sheet_name=None, engine='openpyxl')
        asset_dfs = []
        for sheet_name, sheet_df in asset_sheets.items():
            sheet_df.rename(columns={'DATE': 'date'}, inplace=True)
            asset_dfs.append(sheet_df)
        asset_df = pd.concat(asset_dfs, ignore_index=True)
    st.write("Fichier assets lu avec succès.")

    # Étape 2: Lecture du fichier OPCVM
    if uploaded_opcvm.name.endswith('.csv'):
        opcvm_df = pd.read_csv(uploaded_opcvm)
    else:
        opcvm_df = pd.read_excel(uploaded_opcvm, sheet_name='Sheet1', engine='openpyxl')
    opcvm_df.rename(columns={'Sheet_Date': 'date', '1 jour': '1jour', 'OPCVM': 'opcvm'}, inplace=True)
    st.write("Fichier OPCVM lu avec succès.")

    # Merge des données sur 'date'
    df = pd.merge(opcvm_df[['date', 'opcvm', '1jour']], asset_df, on='date', how='inner')
    st.write("Données mergées avec succès.")

    # Nettoyage: Suppression des lignes avec NaN dans '1jour'
    df.dropna(subset=['1jour'], inplace=True)

    # Identification des colonnes assets (tout sauf 'date', 'opcvm', '1jour')
    asset_columns = [col for col in df.columns if col not in ['date', 'opcvm', '1jour']]

    # Suppression des NaN dans les colonnes assets (pour éviter erreurs dans le modèle)
    df.dropna(subset=asset_columns, inplace=True)

    # Sélection de l'OPCVM (puisque multiple possible)
    unique_opcvm = df['opcvm'].unique()
    selected_opcvm = st.selectbox("Sélectionnez l'OPCVM à analyser", unique_opcvm)

    # Filtrage des données pour l'OPCVM sélectionné
    opcvm_data = df[df['opcvm'] == selected_opcvm].copy()

    # Préparation des features (X) et target (y)
    X = opcvm_data[asset_columns]
    y = opcvm_data['1jour']

    # Vérification si assez de données
    if len(opcvm_data) < 10:
        st.error("Pas assez de données pour cet OPCVM.")
        st.stop()

    # Split train/test (80/20, random pour simplicité, mais idéalement time series split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sélecteur de modèle
    model_choice = st.selectbox("Sélectionnez le modèle Machine Learning",
                                ["Régression Linéaire", "Random Forest Regressor", "Gradient Boosting Regressor"])

    # Instanciation du modèle
    if model_choice == "Régression Linéaire":
        model = LinearRegression()
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)

    # Étape 2: Entraînement du modèle
    model.fit(X_train, y_train)

    # Étape 3: Prédiction sur test
    y_pred = model.predict(X_test)

    # Calcul du R²
    r2 = r2_score(y_test, y_pred)
    st.write(f"Score R² sur les données de test : {r2:.4f}")

    # Étape 4: Résumé des résultats
    if r2 > 0.8:
        quality = "excellente"
    elif r2 > 0.5:
        quality = "bonne"
    else:
        quality = "moyenne ou faible"
    st.write(f"""
    **Résumé :** La qualité de la prédiction est {quality}. 
    Le modèle explique {r2 * 100:.2f}% de la variance des rendements journaliers de l'OPCVM 
    en utilisant les rendements des maturités et titres comme features. 
    Cela donne une estimation de la composition du portefeuille de l'OPCVM.
    """)

    # Étape 5: Affichage de la composition estimée
    # Pour LR: coefficients comme poids
    # Pour RF/GB: feature importances comme proxy d'importance dans la composition
    st.subheader("Composition estimée de l'OPCVM")
    if model_choice == "Régression Linéaire":
        coefficients = pd.DataFrame({'Asset': asset_columns, 'Poids': model.coef_})
        coefficients = coefficients.sort_values('Poids', ascending=False)
        st.dataframe(coefficients)
    else:
        importances = pd.DataFrame({'Asset': asset_columns, 'Importance': model.feature_importances_})
        importances = importances.sort_values('Importance', ascending=False)
        st.dataframe(importances)

    # Préparation des résultats prédits
    results = pd.DataFrame({'Rendement Réel': y_test, 'Rendement Prédit': y_pred})
    # Ajout de la date pour contexte (en utilisant l'index de y_test)
    results['Date'] = opcvm_data.loc[y_test.index, 'date']
    results = results[['Date', 'Rendement Réel', 'Rendement Prédit']]

    # Affichage des résultats
    st.subheader("Résultats Prédits vs Réels sur les Données de Test")
    st.dataframe(results)

    # Étape 6: Option de téléchargement
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les résultats en CSV",
        data=csv,
        file_name="resultats_predits.csv",
        mime="text/csv"
    )