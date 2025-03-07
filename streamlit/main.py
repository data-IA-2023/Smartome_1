import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration du thème
st.set_page_config(page_title="Prédiction du Temps de Chauffe", layout="wide")

# Titre stylisé
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction du Temps de Chauffe</h1>", unsafe_allow_html=True)

# Téléchargement du fichier CSV
st.sidebar.header("1. Téléchargez vos données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV", type=["csv"])

if uploaded_file:
    df1 = pd.read_csv(uploaded_file)
    df = df1.head(5000)
    
    # Afficher les premières lignes et colonnes
    st.write("### Aperçu des données")
    st.write(df.head(10))
    st.write(f"##### Longueur du document: {len(df)} lignes")


    required_columns = ['Adresse', 'Code Postal', 'Ville', 'Datetime', 
                        'Consigne Température (°C)', 'Température Intérieure (°C)',
                        'Température Extérieure (°C)', 'Présence', 'Humidité (%)',
                        'Ensoleillement (h)', 'Orientation', 'DPE Classe', 'DPE Valeur',
                        'Année de fabrication', 'Surface (m²)', 'Surface (m³)', 'Puissance',
                        'temps de chauffe', 'Nombre de pièces']

    missing_cols = [col for col in required_columns if col not in df.columns]

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Année'] = df['Datetime'].dt.year
    df['Mois'] = df['Datetime'].dt.month
    df['Jour'] = df['Datetime'].dt.day
    df['Heure'] = df['Datetime'].dt.hour

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Hiver'
        elif month in [3, 4, 5]:
            return 'Printemps'
        elif month in [6, 7, 8]:
            return 'Été'
        else:
            return 'Automne'
    df['Saison'] = df['Mois'].apply(get_season)
    
    
    # Séparation des colonnes numériques et catégorielles
    numeric_cols = ['Consigne Température (°C)', 'Température Intérieure (°C)', 
                    'Température Extérieure (°C)', 'Humidité (%)', 'Ensoleillement (h)', 
                    'DPE Valeur', 'Puissance','Année', 'Mois', 'Jour', 'Heure', 'temps de chauffe'
                    ]
    cat_cols = ['Saison','Ville', 'Orientation', 'DPE Classe','Surface (m²)', 'Surface (m³)','Nombre de pièces']

    correlation_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matrice de Corrélation")
    st.pyplot(plt)  

    numeric_cols.remove('temps de chauffe')
    df[cat_cols] = df[cat_cols].astype(str)
    df_encoded = pd.get_dummies(df[cat_cols])
   
    X = pd.concat([df[numeric_cols], df_encoded], axis=1)
    y = df['temps de chauffe']
    st.write(X.head(10))
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    models = {
        "Régression Linéaire": (LinearRegression(), {"fit_intercept": [True, False]}),
        "Forêt Aléatoire": (RandomForestRegressor(), {"n_estimators": [50, 100], "max_depth": [10, 20, None]})
    }

    def train_best_model(model_name):
        best_model = None
        best_score = float("inf")
        best_params = {}

        for name, (model, params) in models.items():
            grid = GridSearchCV(model, params, scoring="neg_mean_absolute_error", cv=3)
            grid.fit(X_train, y_train)
            score = -grid.best_score_

            if score < best_score:
                best_score = score
                best_model = grid.best_estimator_
                best_params = grid.best_params_

        st.session_state.saved_models.append({
            "name": model_name,
            "model": best_model,
            "score": best_score,
            "params": best_params
        })
        return best_model, best_score, best_params

    # Initialisation de la session
    if "saved_models" not in st.session_state:
        st.session_state.saved_models = []

    # Entraînement du modèle
    st.sidebar.header("2. Entraînez le modèle")
    model_name_input = st.sidebar.text_input("Nom du modèle", "Mon modèle")

    if st.sidebar.button("Entraîner le modèle"):
        with st.spinner("Entraînement en cours..."):
            best_model, best_score, best_params = train_best_model(model_name_input)
            st.write(f"### Modèle '{model_name_input}' entraîné avec succès.")
            st.write(f"**Paramètres optimaux** : {best_params}")

            # Évaluation
            y_pred = best_model.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)

            st.write(f"**MAE** : {test_mae:.2f}")
            st.write(f"**MSE** : {test_mse:.2f}")
            st.write(f"**R2 Score** : {test_r2:.2f}")

            # Graphiques
            fig_residuals = plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_pred, y=(y_test - y_pred))
            plt.axhline(0, color='red', linestyle='--')
            plt.title("Résidus vs Prédictions")
            plt.xlabel("Prédictions")
            plt.ylabel("Résidus")
            st.pyplot(fig_residuals)

            fig_actual_vs_predicted = plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
            plt.title("Valeurs réelles vs prédites")
            plt.xlabel("Valeurs réelles")
            plt.ylabel("Valeurs prédites")
            st.pyplot(fig_actual_vs_predicted)

            if isinstance(best_model, RandomForestRegressor):
                importances = best_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                features_sorted = np.array(X.columns)[indices]

                fig_importance = plt.figure(figsize=(10, 6))
                plt.barh(features_sorted, importances[indices])     
                plt.title("Importance des Caractéristiques")
                st.pyplot(fig_importance)

    # Sélection et prédiction avec le modèle entraîné
    st.sidebar.header("3. Sélectionner un modèle pour prédiction")

    if "saved_models" in st.session_state and st.session_state.saved_models:
        model_names = [model['name'] for model in st.session_state.saved_models]
        selected_model_name = st.sidebar.selectbox("Choisissez un modèle", model_names)

        selected_model_data = next((model for model in st.session_state.saved_models if model['name'] == selected_model_name), None)

        if selected_model_data:
            selected_model = selected_model_data['model']

            # Formulaire de prédiction
            with st.sidebar.form("user_input_form"):
                user_input = {}
                for feature in numeric_cols:
                    user_input[feature] = st.number_input(f"{feature}", value=0.0)

                for feature in cat_cols:
                    user_input[feature] = st.selectbox(f"{feature}", df[feature].unique().tolist())

                submit_button = st.form_submit_button("Prédire")

            if submit_button:
                input_df = pd.DataFrame([user_input])
                input_df_encoded = pd.get_dummies(input_df, drop_first=True)
                input_df_encoded = input_df_encoded.reindex(columns=X_train.columns, fill_value=0)

                prediction = selected_model.predict(input_df_encoded)
                st.sidebar.write(f"### Temps de chauffe prédit : {prediction[0]:.0f} minutes")
        else:
            st.sidebar.warning("Veuillez entraîner un modèle avant de faire une prédiction.")
    else:
        st.sidebar.info("Aucun modèle enregistré. Veuillez d'abord entraîner un modèle.")
