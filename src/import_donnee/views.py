##########--Django--################
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib import messages

##########--Base de données--################
from utils import get_db_mongo

##########--Modélisation--################
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,MinMaxScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging
import warnings

##########--Visualisation--################
import matplotlib.pyplot as plt
import seaborn as sns
import base64,io

##########--Autre--################
import pandas as pd
import json
from .tasks import train_model_task




# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



@login_required
def accueil(request):
    col, client = get_db_mongo()
    user_id = str(request.user.id)
    user_doc = col.find_one({'_id': user_id})
    project_names = [project['name'] for project in user_doc.get('projects', [])] if user_doc else []
    client.close()
    
    return render(request, 'accueil.html', {'project_names': project_names})


@login_required
def create_project(request):
    if request.method == 'POST':
        col, client = get_db_mongo()
        project_name = request.POST.get('project_name')
        user_id = str(request.user.id)
        user_name = request.user.username
        user_doc = col.find_one({'_id': user_id})
        if not user_doc:
            print(f"Utilisateur {user_id} non trouvé, création d'un nouveau document")
            new_user = {
                '_id': user_id,
                'username': user_name,
                'projects': []
            }
            col.insert_one(new_user)
            user_doc = new_user  
        existing_project = next((project for project in user_doc['projects'] if project['name'] == project_name), None)
        if existing_project:
            messages.error(request, f"Le projet '{project_name}' existe déjà.")
            return redirect('accueil')
        new_project = {  
            'name': project_name,
            'data': []
        }
        col.update_one(
            {'_id': user_id},
            {'$push': {'projects': new_project}}
        )
        client.close()
        messages.success(request, f"Le projet '{project_name}' a été créé avec succès.")
        return redirect('accueil')

    return render(request, 'accueil.html')

@login_required
def delete_project(request):
    if request.method == 'POST':
        col, client = get_db_mongo()
        project_name = request.POST.get('project_name')
        user_id = str(request.user.id)
        col.update_one(
            {'_id': user_id},
            {'$pull': {'projects': {'name': project_name}}})
        client.close()
        messages.success(request, f"Le projet '{project_name}' a été supprimé avec succès.")
        return redirect('accueil')


@login_required
def projects(request):
    col, client = get_db_mongo()
    user_id = str(request.user.id)
    user_doc = col.find_one({'_id': user_id})
    if user_doc:
        projects = user_doc.get('projects', [])
    else:
        projects = []
    client.close()
    return render(request, 'projet.html', {'projects': projects})


@login_required
def projet_data(request):
    if request.method == 'POST':  
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        project_name = request.POST.get('projet_name')  
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        client.close()
        dataset_names = [dataset['dataset_name'] for dataset in project.get('data', [])]
        return render(request, 'projet_dataset.html',{'project_name':project_name,"dataset_names":dataset_names})
    return redirect('upload_fichier')  




@login_required
def upload_fichier(request):
    if request.method == 'POST':
        project_name = request.POST.get('projet_name')
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        dataset_names = [dataset['dataset_name'] for dataset in project.get('data', [])]
        separator = request.POST.get('separator', ',')  
        uploaded_file = request.FILES.get('file')
        if uploaded_file:
            file_size = uploaded_file.size
            if file_size > 15 * 1024 * 1024:  
                messages.error(request, "La taille du fichier ne doit pas dépasser 5 Mo.")
                return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, sep=separator)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                elif uploaded_file.name.endswith('.xls'):
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                else:
                    messages.error(request, "Format de fichier non supporté.")
                    return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})
                data_dict = df.to_dict("records")
                if not data_dict:
                    messages.error(request, "Aucune donnée n'a été lue à partir du fichier.")
                    return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})
                dataset_name = uploaded_file.name  
                existing_dataset = next((dataset for dataset in project.get('data', []) if dataset['dataset_name'] == dataset_name), None)
                if existing_dataset:
                    messages.error(request, "Un dataset avec ce nom existe déjà dans ce projet.")
                    return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})
                organized_data = {
                    'dataset_name': dataset_name,
                    'data': data_dict,
                    'graphs':[]
                }
                col.update_one(
                    {'_id': user_id, 'projects.name': project_name},
                    {'$push': {'projects.$.data': organized_data}}
                )
                user_doc = col.find_one({'_id': user_id})
                projects = user_doc.get('projects', [])
                project = next((p for p in projects if p['name'] == project_name), None)
                dataset_names = [dataset['dataset_name'] for dataset in project.get('data', [])]
                client.close()
                messages.success(request, f"Le fichier a été téléchargé avec succès dans le projet {project_name}.")
                return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})
            except Exception as e:
                messages.error(request, f"Erreur lors du traitement du fichier : {str(e)}")
                return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})
        else:
            messages.error(request, "Aucun fichier n'a été téléchargé.")
            return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})

    return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names})




@login_required
def dataset_info(request):
    if request.method == "POST":
        project_name = request.POST.get('project_name')
        dataset_name = request.POST.get('selected_dataset')
        action = request.POST.get('action')
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        dataset_names = [dataset['dataset_name'] for dataset in project.get('data', [])]
        dataset = next((d for d in project.get('data', []) if d['dataset_name'] == dataset_name), None)
        saved_graphs = dataset.get('graphs', [])
        data = dataset.get('data', [])
        df = pd.DataFrame(data)
        columns = df.columns
        column_types = df.dtypes.apply(lambda x: x.name).to_dict()
        columns_with_types = [(col, column_types[col]) for col in columns]
        object_columns = [col for col, col_type in columns_with_types if col_type == "object"]
        numeric_columns = [col for col, col_type in df.dtypes.items() if col_type in ['int64', 'float64']]
        encoding_methods_numerique = [("robust", "RobustScaler"), ("standard", "StandardScaler"),("minmax","MinMaxScaler")]
        encoding_methods = [("onehot", "One-Hot Encoding"), ("label", "Label Encoding")]
        outlier_info = calculate_outliers_zscore(df,numeric_columns)
        if action == "action1" or action == "action5":
            try:
                df = pd.DataFrame(dataset['data'])
                ligne = df.shape[0]  
                colonne = df.shape[1]  
                
                print("outlier_info",outlier_info)
                nb_nul = df.isnull().sum().to_frame(name="Nombre de valeurs nulles").to_html(classes="table table-bordered") 
                nb_colonne_double = df.duplicated().sum()  
                table_html=df.to_html(classes='display',table_id="dataframe-table",index=False)
                client.close()
                return render(request, 'projet_dataset.html', {'table_html':table_html,
                    'dataset_name': dataset_name,
                    "dataset_names":dataset_names,
                    "outlier_info":outlier_info,
                    'ligne': ligne,
                    'colonne': colonne,
                    'nb_nul': nb_nul,
                    'nb_colonne_double': nb_colonne_double,
                    "project_name":project_name,
                    "numeric_columns":numeric_columns,
                })
            except Exception as e:
                messages.error(request, f"Erreur lors de l'analyse du dataset : {str(e)}")
                client.close()
                return render(request,'projet_dataset.html',{"project_name":project_name,'dataset_name': dataset_name,'dataset_names':dataset_names})
        elif action == "action2":
            return render(request,'nettoyage.html',{"encoding_methods_numerique":encoding_methods_numerique,
                                                    "outlier_info":outlier_info,
                                                    "encoding_methods":encoding_methods,
                                                    "numeric_columns":numeric_columns,
                                                    "object_columns":object_columns,   
                                                    "columns_with_types":columns_with_types,
                                                    "columns":columns,"project_name":project_name,
                                                    "dataset_name":dataset_name,
                                                    'dataset_names':dataset_names})
        elif action == "action3":
            col.update_one(
            {'_id': user_id, 'projects.name': project_name},
            {'$pull': {'projects.$.data': {'dataset_name': dataset_name}}})
            user_doc = col.find_one({'_id': user_id})
            projects = user_doc.get('projects', [])
            project = next((p for p in projects if p['name'] == project_name), None)
            dataset_names = [dataset['dataset_name'] for dataset in project.get('data', [])]
            client.close()
            messages.success(request, f"Le dataset '{dataset_name}' a été supprimé avec succès.")
            return render(request, 'projet_dataset.html', {"project_name": project_name, "dataset_names": dataset_names})
        elif action == "action4":
            return render(request,'visualisation.html',{"saved_graphs":saved_graphs,"numeric_columns":numeric_columns,"project_name":project_name,'dataset_name': dataset_name})
        elif action == "action6":
            models = {
            "random_forest": RandomForestClassifier(),
            "svc": SVC(),
            "logistic_regression": LogisticRegression()
            }

            # Grilles de recherche des hyperparamètres
            param_grid = {
                "random_forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
                "svc": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                "logistic_regression": {"C": [0.1, 1, 10], "solver": ["liblinear"]}
            }
            return render(request,'modelisation.html',{"models":models,'param_grid': json.dumps(param_grid),"columns":columns,"project_name":project_name,'dataset_name': dataset_name})    
    else:
        return render(request,'projet_dataset.html',{"project_name":project_name,'dataset_name': dataset_name,'dataset_names':dataset_names})
    

@login_required
def visualisation(request):
    if request.method == "POST":
        project_name = request.POST.get('project_name')
        dataset_name = request.POST.get('dataset_name')
        action = request.POST.get('action')
        column_fig = request.POST.getlist('columns', None)
        selected_graphs = request.POST.getlist('graphiques',None)
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        dataset = next((d for d in project.get('data', []) if d['dataset_name'] == dataset_name), None)
        saved_graphs = dataset.get('graphs', [])
        graph_name = request.POST.get('graph_name',None)
        graph_data = request.POST.get('graph_data',None)
        graph_to_delete = request.POST.get('graph_name_to_delete',None)
        data = dataset.get('data', [])
        df = pd.DataFrame(data)
        columns = df.columns
        column_types = df.dtypes.apply(lambda x: x.name).to_dict()
        columns_with_types = [(col, column_types[col]) for col in columns]
        object_columns = [col for col, col_type in columns_with_types if col_type == "object"]
        numeric_columns = [col for col, col_type in df.dtypes.items() if col_type in ['int64', 'float64']]
        if "all" in column_fig:
            column_fig = numeric_columns
        scat=None
        heatmap=None
        graph_2=None
        graph_1=None
        graphs_to_display=None
        if action== "action1":
           try:
                graph_1,graph_2 = visu_1d(df,column_fig)
                messages.success(request, f"Voici le graphique avec les colonnes:{column_fig}")
           except:
               messages.error(request, f"Impossible de générer le graphique avec les colonnes:{column_fig}")
               return render(request,"visualisation.html",{"project_name":project_name,
                                                'dataset_name': dataset_name,
                                                "numeric_columns":numeric_columns,
                                                "action":action,
                                                })
        if action== "action2":
           try:
                scat, heatmap = generate_plots(df[numeric_columns],column_fig[0],column_fig[1])
                messages.success(request, f"Voici le graphique avec les colonnes:{column_fig}")
           except:
               messages.error(request, f"Impossible de générer le graphique avec les colonnes:{column_fig}")
               return render(request,"visualisation.html",{"project_name":project_name,
                                                'dataset_name': dataset_name,
                                                "numeric_columns":numeric_columns,
                                                "action":action,
                                                })
        if action== "action3":
            try:
                save_graph_to_dataset(request,graph_data,graph_name,project_name,dataset_name)
                
            except:
                messages.error(request, f"Impossible de sauvegarder le graphique :{graph_name}")

        if action == "action4":
            graphs_to_display = [
                    (graph['name'], base64.b64encode(graph['image_data']).decode('utf-8')) 
                    for graph in saved_graphs if graph['name'] in selected_graphs
                ]
        if action == 'action5':
             dataset['graphs'] = [graph for graph in saved_graphs if graph['name'] != graph_to_delete]
             col.update_one({'_id': user_id}, {"$set": {"projects": projects}})
             messages.success(request, f"Graphique '{graph_to_delete}' supprimé avec succès.")
        
            
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        dataset = next((d for d in project.get('data', []) if d['dataset_name'] == dataset_name), None)
        saved_graphs = dataset.get('graphs', [])  
        return render(request,"visualisation.html",{"project_name":project_name,
                                                'dataset_name': dataset_name,
                                                "graphs_to_display":graphs_to_display,
                                                "saved_graphs":saved_graphs,
                                                'graph_2': graph_2,
                                                'graph_1': graph_1,
                                                'scat':scat,
                                                'heatmap':heatmap,
                                                "numeric_columns":numeric_columns,
                                                 "action":action,
                                                })
    

@login_required
def modelisation(request):
    if request.method == "POST":
        project_name = request.POST.get('project_name')
        dataset_name = request.POST.get('dataset_name')
        features = request.POST.getlist('features')
        target = request.POST.get('target')
        model_name = request.POST.get('model',None)
        model_selection = request.POST.get('model_selection',None)
        param_mode = request.POST.get('param_mode')
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        dataset = next((d for d in project.get('data', []) if d['dataset_name'] == dataset_name), None)
        data = dataset.get('data', [])
        df = pd.DataFrame(data)
        columns = df.columns
        print("model_name",model_name)
        # Dictionnaire des modèles disponibles
        models = {
            'random_forest': RandomForestClassifier(),
            'svc': SVC(),
            'logistic_regression': LogisticRegression()
        }

        # Paramètres de GridSearch (facultatif)
        param_grid = {
            'random_forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]},
            'svc': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'logistic_regression': {'C': [0.1, 1, 10]}
        }
        manual_params = {}
        use_grid_search=False
        if model_selection == "grid_search":
            search_models = True
        if model_selection == "manual_model":
            search_models=False
        if param_mode == 'manual':
            for param in request.POST:
                if param not in ['project_name', 'dataset_name', 'target', 'features', 'model', 'param_mode','csrfmiddlewaretoken','model_selection']:
                    manual_params[param] = request.POST.get(param)
        # Séparation des données en X (features) et y (target)
        if param_mode =='auto':
            use_grid_search = True

        if target in features:
            features.remove(target)
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
            messages.success(request, "Données séparées avec succès!")
        except Exception as e:
            messages.error(request, f"Erreur lors de la séparation des données: {e}")
        # Entraînement du modèle
        try:
            model, best_params, message = train_model(
                X_train, 
                Y_train, 
                model_name=model_name, 
                params=manual_params, 
                use_grid_search=use_grid_search,  
                search_models=search_models,  
                scoring=["accuracy", "roc_auc", "f1"],  # Choisissez la métrique que vous souhaitez utiliser
                cv=10,  # Nombre de folds pour la validation croisée
                models=models, 
                param_grid=param_grid
            )
             # Affichage des résultats de GridSearchCV si applicable
            if isinstance(model, GridSearchCV):
                print("\n🔍 Meilleurs scores par métrique dans GridSearchCV :")
                for metric in ["accuracy", "roc_auc", "f1"]:
                    print(f"  {metric}: {max(model.cv_results_['mean_test_' + metric])}")

                print("\n🎯 Meilleurs paramètres trouvés :")
                print(model.best_params_)

            # Prédictions sur les données de test
            y_pred = model.predict(X_test)

            # Calcul des métriques sur le jeu de test
            accuracy = accuracy_score(Y_test, y_pred)
            messages.success(request, f"Précision sur le jeu de test: {accuracy:.4f}")

            print("\n📊 Rapport de classification sur le jeu de test :")
            print(classification_report(Y_test, y_pred))
            messages.success(request, message)
        except Exception as e:
            messages.error(request, f"Erreur lors de l'entraînement du modèle: {e}")

        # Calcul de la précision
        try:
            # Prédictions sur les données de test
            y_pred = model.predict(X_test)

            # Calcul de l'accuracy
            accuracy = accuracy_score(Y_test, y_pred)
            messages.success(request, f"Précision: {accuracy}")
            
            # Affichage du modèle et de ses meilleurs paramètres
            print("Modèle:", model)
            
            # Récupérer et afficher les paramètres du modèle (après GridSearch ou par défaut)
            print("Paramètres du modèle (après GridSearch ou par défaut):")
            print(model.get_params())

            # Si c'est un modèle GridSearchCV, tu peux aussi afficher les meilleurs paramètres trouvés
            if isinstance(model, GridSearchCV):
                print("Meilleurs paramètres trouvés par GridSearchCV:")
                print(model.best_params_)

            # Affichage de l'accuracy
            print("Accuracy:", accuracy)
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {e}")
        
    
        
       
        return render(request, "modelisation.html", {'param_grid': json.dumps(param_grid),"models":models,"project_name": project_name, "dataset_name": dataset_name, "columns": columns})

@login_required
def cleanning(request):
     if request.method == "POST":
        project_name = request.POST.get('project_name')
        dataset_name = request.POST.get('dataset_name',None)
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        dataset_names = [dataset['dataset_name'] for dataset in project.get('data', [])]
        return render(request,'projet_dataset.html',{"project_name":project_name,"dataset_names":dataset_names,"dataset_name":dataset_name})


@login_required
def imputation(request):
    if request.method == "POST":
        project_name = request.POST.get('project_name')
        dataset_name = request.POST.get('dataset_name')
        column = request.POST.get('column',None)
        strategy = request.POST.get('strategy', None)
        action = request.POST.get('action')
        replace_value = request.POST.get('replace_value', None)
        old_value = request.POST.get('old_value', None)
        new_value = request.POST.get('new_value', None)
        selected_columns_encod_cat = request.POST.getlist("columns_cat",None)
        selected_columns_encod_num = request.POST.getlist("columns_num",None)
        encoding_method = request.POST.get("encoding_method", None)
        # print("project_name",project_name)
        # print("dataset_name",dataset_name)
        # print("column",column)

        # print("strategy",strategy)
        # print("action",action)
        # print("replace_value",replace_value)
        # print("old_value",old_value)
        # print("new_value",new_value)
        # print("selected_columns_encod_cat",selected_columns_encod_cat)
        # print("selected_columns_encod_num",selected_columns_encod_num)
        # print("encoding_method",encoding_method)
        # print("columns",columns)
        # print("column_types",column_types)
        # print("columns_with_types",columns_with_types)
        # print("numeric_columns",numeric_columns)
        # print("object_columns",object_columns)
        # print("encoding_methods",encoding_methods)
        col, client = get_db_mongo()
        user_id = str(request.user.id)
        user_doc = col.find_one({'_id': user_id})
        projects = user_doc.get('projects', [])
        project = next((p for p in projects if p['name'] == project_name), None)
        datasets = project.get('data', [])
        dataset = next((d for d in datasets if d['dataset_name'] == dataset_name), None)
        data = dataset.get('data', [])
        df = pd.DataFrame(data)
        columns = df.columns
        column_types = df.dtypes.apply(lambda x: x.name).to_dict()
        columns_with_types = [(col, column_types[col]) for col in columns]
        object_columns = [col for col, col_type in columns_with_types if col_type == "object"]
        numeric_columns = [col for col, col_type in df.dtypes.items() if col_type in ['int64', 'float64']]
        encoding_methods = [("onehot", "One-Hot Encoding"), ("label", "Label Encoding")]
        encoding_methods_numerique = [("robust", "RobustScaler"), ("standard", "StandardScaler"),("minmax","MinMaxScaler")]
        
        if action == "action1":
            df = drop_doublons(df)
            messages.success(request, f"Les doublons ont été supprimés avec succès.")
        if action == "action2":
            df = valeurs_manquantes(df, column, strategy,replace_value)
            messages.success(request, f"Les valeurs manquantes ont été imputées avec succès.")
        if action == "action3":
            df,message = replace(df,column,old_value,new_value)
            messages.success(request, message)
        if action == "action4":
            df,message = encode_categorical(df, selected_columns_encod_cat, encoding_method)
            messages.success(request, message)
        if action == "action5":
            df,message = encode_numeric(df, selected_columns_encod_num, encoding_method)
            messages.success(request, message)
            
            
            
        col.update_one(
            {'_id': user_id, 'projects.name': project_name, 'projects.data.dataset_name': dataset_name},
            {'$set': {'projects.$.data.$[elem].data': df.to_dict('records')}},
            array_filters=[{"elem.dataset_name": dataset_name}]
        )

        
        client.close()
        return render(request, 'nettoyage.html', {"encoding_methods_numerique":encoding_methods_numerique,
                                                  "encoding_methods":encoding_methods,
                                                  "object_columns":object_columns,
                                                  "numeric_columns":numeric_columns,
                                                  "project_name": project_name, 
                                                  "dataset_name": dataset_name, 
                                                  "columns_with_types": columns_with_types,
                                                  "action":action})






#####################################################Traitement################################################################
def calculate_outliers_zscore(df, numeric_columns, threshold=2):
    outlier_info = []

    for column in numeric_columns:
        # Ignorer les valeurs manquantes
        data = df[column].dropna()

        # Calcul de la moyenne et de l'écart-type
        mean = data.mean()
        std_dev = data.std()

        # Vérification de l'écart-type
        if std_dev == 0:
            print(f"Colonne {column}: Toutes les valeurs sont identiques. Pas d'outliers possibles.")
            outlier_info.append({
                'column': column,
                'outliers': [],
                'outliers_count': 0,
                'outlier_indices': []  # Aucune ligne d'outlier
            })
            continue

        # Calcul des Z-scores
        z_scores = (data - mean) / std_dev

        # Filtrage des outliers
        outliers = data[z_scores.abs() > threshold]

        # Récupération des indices des outliers
        outlier_indices = outliers.index.tolist()

        # Comptage des outliers
        outliers_count = len(outlier_indices)

        # Ajouter les informations des outliers dans la liste
        outlier_info.append({
            'column': column,
            'outliers_count': outliers_count,
            "outlier_indices":outlier_indices
        })

    return outlier_info


def valeurs_manquantes(df, col, missing_strategy,replace_value):
    if missing_strategy == "mean":
        df[col].fillna(df[col].mean(), inplace=True)
    elif missing_strategy == "median":
        df[col].fillna(df[col].median(), inplace=True)
    elif missing_strategy == "mode":
        df[col].fillna(df[col].mode()[0], inplace=True)
    elif missing_strategy == "drop":
        df.dropna(subset=[col], inplace=True)
    elif missing_strategy == "replace" and replace_value is not None:
        df[col].fillna(replace_value, inplace=True)
    return df


def drop_doublons(df):
    df =df.drop_duplicates()
    return df

def magic_clean(df):
    """
    Nettoie et prépare les colonnes de type 'object' dans un DataFrame.
    """
    for col in df.select_dtypes(include=['object']).columns:
        # Supprime les caractères non imprimables et les espaces inutiles
        df[col] = df[col].astype(str).str.replace(r'[\s\x00-\x1F\x7F-\x9F]+', '', regex=True)

        # Remplace les virgules par des points pour les conversions numériques
        df[col] = df[col].str.replace(',', '.', regex=False)

        # Tente de convertir les colonnes en numérique
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            if converted.notna().all():
                # Conversion en int si toutes les valeurs sont des entiers, sinon en float
                try:
                    df[col] = converted.astype(int)
                except ValueError:
                    df[col] = converted.astype(float)
        except ValueError:
            # Vérifie si la colonne contient des dictionnaires ou des listes
            is_dict = df[col].apply(lambda x: isinstance(x, dict)).all()
            is_list = df[col].apply(lambda x: isinstance(x, list)).all()

            if not (is_dict or is_list):
                # Si ce n'est ni une liste ni un dictionnaire, force en chaîne
                df[col] = df[col].astype(str)
    return df

def replace(df, col, old_value, new_value):
    magic_clean(df)
    col_type = df[col].dtype
    try:
        if pd.api.types.is_numeric_dtype(col_type):
            old_value = float(old_value) if '.' in str(old_value) else int(old_value)
        elif pd.api.types.is_datetime64_any_dtype(col_type):
            old_value = pd.to_datetime(old_value)
        elif pd.api.types.is_bool_dtype(col_type):
            old_value = bool(old_value)
        else:
            old_value = str(old_value)
    except (ValueError, TypeError):
        message = f"Impossible de convertir les valeurs '{old_value}' et '{new_value}' pour correspondre au type '{col_type}'."
    # Vérifie si la valeur à remplacer existe dans la colonne
    if old_value not in df[col].values:
        message =  f"La valeur '{old_value}' n'est pas présente dans la colonne '{col}'."
    else:
        df[col] = df[col].replace(old_value, new_value)
        message = f"Remplacement de '{old_value}' par '{new_value}' effectué avec succès dans la colonne '{col}'."
    return df, message


def encode_categorical(df, col, encoding_method):
 
    if isinstance(col, str):
        col = [col]  
    try:
        if encoding_method == "onehot":
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(df[col])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(col))
            df = df.drop(columns=col).reset_index(drop=True)
            df = pd.concat([df, encoded_df], axis=1)
            message = f"Encodage {encoder.__class__.__name__} effectué avec succès."

        elif encoding_method == "label":
            for column in col:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                message = f"Encodage {le.__class__.__name__} effectué avec succès."
    except Exception as e:
        message = f"Erreur lors de l'encodage : {str(e)}"
        return df, message
    return df, message
        

def encode_numeric(df,col,encoding_method):
    magic_clean(df)
    print(encoding_method)
    print(col)
    if isinstance(col, str):
        col = [col]
    if encoding_method == "standard":
        scaler = StandardScaler() 
    elif encoding_method == "minmax":
        scaler = MinMaxScaler()
    elif encoding_method == "robust":
        scaler = RobustScaler()
    try:
        encoded = scaler.fit_transform(df[col])
        encoded_df = pd.DataFrame(encoded, columns=scaler.get_feature_names_out(col))
        df = df.drop(columns=col).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
    except Exception as e:
        return df ,f'Erreur de l encodage avec {scaler.__class__.__name__}:{e}'
    return df ,f"Encodage {scaler.__class__.__name__} effectué avec succès."

#####################################################Visualisation################################################################
def save_graph_to_dataset(request,graph_data,graph_name,project_name,dataset_name):
        try:
            image_data = base64.b64decode(graph_data)
            # Récupérer la collection MongoDB
            col, client = get_db_mongo()
            user_id = str(request.user.id)
            user_doc = col.find_one({
            '_id': user_id,
            'projects.name': project_name,
            'projects.data.dataset_name': dataset_name},
            {
            'projects.$': 1  # Limiter la projection aux projets correspondants
            })

            dataset = next(
                (d for d in user_doc['projects'][0]['data'] if d['dataset_name'] == dataset_name),
                None
            )
            # Vérifier si un graphique avec le même nom existe
            if any(graph.get('name') == graph_name for graph in dataset.get('graphs', [])):
                messages.error(request, f"Un graphique nommé '{graph_name}' existe déjà pour le dataset '{dataset_name}'.")
                return

            # Mettre à jour le dataset pour inclure le graphique
            result = col.update_one(
                {
                    '_id': user_id,
                    'projects.name': project_name,
                    'projects.data.dataset_name': dataset_name
                },
                {
                    '$push': {
                        'projects.$[project].data.$[dataset].graphs': {
                            'name': graph_name,
                            'image_data': image_data
                            
                        }
                    }
                },
                array_filters=[
                    {'project.name': project_name},
                    {'dataset.dataset_name': dataset_name}
                ]
            )

            if result.modified_count > 0:
                messages.success(request, f"Le graphique '{graph_name}' a été enregistré avec succès pour le dataset '{dataset_name}' du projet '{project_name}'.")
            else:
                messages.error(request, "Aucune mise à jour effectuée. Vérifiez que le projet et le dataset existent.")
        except Exception as e:
            messages.error(request, f"Erreur lors de l'enregistrement du graphique : {str(e)}")

        return 



def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_str




def visu_1d(df, liste_colonnes_num):
    # Vérifier que la liste des colonnes est bien une liste
    if not isinstance(liste_colonnes_num, list):
        liste_colonnes_num = [liste_colonnes_num]

    # Création du boxplot
    fig_1, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot([df[col].dropna().values for col in liste_colonnes_num], labels=liste_colonnes_num, vert=True)
    ax.set_xlabel("Colonnes")
    ax.set_ylabel("Valeurs")
    plt.grid()

    # Création de l'histogramme
    fig_2, ax = plt.subplots(figsize=(10, 7))
    for col in liste_colonnes_num:
        ax.hist(df[col].dropna().values, bins=5, alpha=0.7, label=col)
    ax.set_xlabel("Valeurs")
    ax.set_ylabel("Fréquence")
    ax.legend(title="Colonnes")
    plt.grid()
    # Retourner les images en base64
    fig1 = fig_to_base64(fig_1)
    fig2 = fig_to_base64(fig_2)
    return fig1, fig2


def generate_plots(df,colonne_x, colonne_y):
    
    # Scatterplot
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.scatter(x=df[colonne_x], y=df[colonne_y], alpha=1, edgecolors='w', c='red')
    ax1.set_xlabel(colonne_x)
    ax1.set_ylabel(colonne_y)
    ax1.set_title(f"Scatterplot de {colonne_x} vs {colonne_y}")
    plt.grid()

    # Heatmap de la matrice de corrélation
    fig2 = plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
    # Convertir les deux figures en Base64
    scat = fig_to_base64(fig1)
    heatmap = fig_to_base64(fig2)

    return scat, heatmap


############################################Machin learning#########################################################



def train_model(X_train, y_train, model_name="random_forest", params=None, use_grid_search=False, search_models=False, scoring="accuracy", cv=5,models = None,param_grid= None):
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
