
from celery import shared_task
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
import logging
 


@shared_task(bind=True)
def train_model_task(self,X_train, y_train, model_name="random_forest", params=None, use_grid_search=False, search_models=False, scoring="accuracy", cv=5,models = None,param_grid= None):
    """
    Entraîne un modèle de Machine Learning avec ou sans GridSearch, en gérant les erreurs potentielles.

    :param X_train: Features d'entraînement
    :param y_train: Labels d'entraînement
    :param model_name: Nom du modèle à utiliser (si search_models=False)
    :param params: Dictionnaire des paramètres si non None et use_grid_search=False
    :param use_grid_search: Si True, utilise GridSearch pour chercher les meilleurs hyperparamètres
    :param search_models: Si True, GridSearch explore aussi plusieurs modèles
    :param scoring: Métrique de scoring pour GridSearch (ex: "accuracy", "f1", "roc_auc", etc.)
    :param cv: Nombre de folds pour la validation croisée
    :return: Modèle entraîné, meilleurs paramètres trouvés (ou ceux par défaut), message d'information
    """
    logging.info("🚀 Début de l'entraînement du modèle...")
    print("modele_nameintofonction",model_name)
    if params:
        for param, value in params.items():
            # Si la valeur est une chaîne qui représente un nombre, on la convertit
            if value.replace('.', '', 1).isdigit():  # Si la valeur peut être un nombre
                params[param] = float(value) if '.' in value else int(value)
    

    

    # Variables pour stocker le meilleur modèle en cas de recherche multiple
    best_model = None
    best_score = float('-inf')
    best_params = None
    models_tested = 0  

    # Désactiver les warnings inutiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Recherche du meilleur modèle parmi plusieurs algorithmes
        if search_models:
            logging.info("🔍 Recherche du meilleur modèle parmi plusieurs algorithmes...")
            
            for model_key, model in models.items():
                try:
                    logging.info(f"🔍 Optimisation du modèle: {model_key} avec scoring={scoring} et cv={cv}")
                    
                    grid_search = GridSearchCV(model, param_grid[model_key], cv=cv, scoring=scoring, refit='accuracy')
                    grid_search.fit(X_train, y_train)
                    
                    models_tested += 1  # Compter les modèles testés avec succès

                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_

                except Exception as e:
                    logging.warning(f"⚠️ Erreur avec le modèle {model_key}: {e}")

            if best_model is None:
                return None, None, "❌ Aucun modèle n'a réussi à s'entraîner."
            else:
                print("Meilleurs scores pour chaque métrique :")
                for metric in scoring:
                    print(f"{metric}: {grid_search.cv_results_['mean_test_' + metric].max()}")
                return best_model, best_params, f"✅ Meilleur modèle trouvé: {best_model} avec score={best_score}"

        # Entraînement d'un seul modèle spécifique
        else:
            model = models[model_name]

            if model is None:
                return None, None, f"❌ Modèle '{model_name}' non reconnu. Choisissez parmi {list(models.keys())}."

            try:
                # Si GridSearch est activé, recherche des meilleurs hyperparamètres
                if use_grid_search:
                    logging.info(f"🔍 Recherche des meilleurs paramètres pour {model_name}...")
                    grid_search = GridSearchCV(model, param_grid[model_name], cv=cv, scoring=scoring, refit='accuracy')
                    grid_search.fit(X_train, y_train)

                    print("Meilleurs scores pour chaque métrique :")
                    for metric in scoring:
                        print(f"{metric}: {grid_search.cv_results_['mean_test_' + metric].max()}")
                    return grid_search.best_estimator_, grid_search.best_params_, f"✅ Modèle optimisé: {model_name}"

                # Si des paramètres sont fournis, les appliquer au modèle
                elif params:
                    model.set_params(**params)

                # Entraînement du modèle
                logging.info(f"🚀 Entraînement du modèle {model_name}...")
                model.fit(X_train, y_train)

                # Retourner les paramètres définis ou ceux par défaut
                return model, params if params else model.get_params(), f"✅ Modèle entraîné: {model_name}"

            except Exception as e:
                return None, None, f"⚠️ Erreur avec {model_name}: {e}"
