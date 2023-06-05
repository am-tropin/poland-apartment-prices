#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from datetime import date #, datetime, timedelta
import itertools

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

# import pickle
# from mlflow import log_metric, log_param, log_artifact
from mlflow.models.signature import infer_signature

# from IPython import get_ipython

# In[ ]:





# In[ ]:


def load_data():
    price_df4 = pd.read_csv("poland_apartments_completed.csv")
    cat_features = ['city', 'district'] # 'floor', 'rooms'?
    num_features = ['floor', 'rooms', 'sq', 'year', 'radius']
    target = ['price']
    return price_df4, cat_features, num_features, target


# In[ ]:





# # 1. For model evaluation

# In[ ]:


def to_split_and_scale(df, cat_features, num_features, target):
    X = df[cat_features + num_features]
    y = df[target]

    # encoding the categorical variables into numerical variables
    labels_dict = {}
    le = LabelEncoder()
    for col in cat_features:
        X[col] = le.fit_transform(X[col])
        labels_dict[col] = dict(zip(le.classes_, range(len(le.classes_))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    st_scaler = StandardScaler()
    X_train_scaled = st_scaler.fit_transform(X_train)
    X_test_scaled = st_scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, st_scaler, labels_dict


# In[ ]:


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# In[ ]:


def reg_model_implementation(model, grid, X_train, y_train, X_test, y_test):
    '''
    This function:
    - fits and returns regression model with GridSearchCV;
    - prints confusion matrix and classification report.
    '''
    
    start_time = time.time()
    mod = model()
    mod_cv = GridSearchCV(mod, grid, cv=10)
    mod_cv.fit(X_train, y_train)
    
    print("Model: {0}".format(namestr(model, globals())[0]))
    if grid != {}:
        print("Tuned hyperparameters:") # , mod_cv.best_params_
        for k, v in mod_cv.best_params_.items():
            print("\t{0}: {1}".format(k, v))
    print()

    mod = model(**mod_cv.best_params_)
    mod.fit(X_train, y_train)
    y_pred_mod = mod.predict(X_test)

    print("R2-score:  ", mod.score(X_test, y_test))
    print("RMSE:      ", mean_squared_error(y_test, y_pred_mod, squared=False))
    print()
    print("Time using clf_model_implementation(): {0:.4f} sec".format(time.time() - start_time))
    
    return mod


# In[ ]:


def show_feature_importances(model, col_names): # list(X_train) !!!!!!! 
    '''
    This function plots the feature importances for given model.
    '''
    resultdict = {}
    importance = model.feature_importances_

    print("score:\t  feature:")
    for i in range(len(col_names)):
        resultdict[col_names[i]] = importance[i]
    
    resultdict = dict(sorted(resultdict.items(), key=lambda item: -item[1]))
    for k, v in resultdict.items():
        print("{1:.3f}\t- {0}".format(k, v))

    plt.bar(resultdict.keys(),resultdict.values())
    plt.xticks(rotation='vertical')
    plt.title('Feature Importance')
    plt.show()
    
    return 1


# # 2. For MLFlow tracking

# In[ ]:


def rmsle_cv_score(model, X_train, y_train):
    kf = KFold(n_splits=3, shuffle=True, random_state=42).get_n_splits(X_train)
    return np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))


# In[ ]:


# Track params and metrics
def log_mlflow_run(model, signature, parameters, metrics):
    # Auto-logging for scikit-learn estimators
    # mlflow.sklearn.autolog()

    # log estimator_name name
    mlflow.set_tag("estimator_name", model.__class__.__name__)

    # log input features
#     mlflow.set_tag("features", str(X_train_scaled.tolist())) # X_train_scaled.columns.values.tolist()

    # Log tracked parameters only
    mlflow.log_params({key: model.get_params()[key] for key in parameters})

    mlflow.log_metrics(metrics)

    # log training loss (ONLY FOR GradientBoostingRegressor ???)
#     for s in model.train_score_:
#         mlflow.log_metric("Train Loss", s)

    # Save model to artifacts
    mlflow.sklearn.log_model(model, "model", signature=signature)

    # log charts
#     mlflow.log_artifacts("model_artifacts")


# In[ ]:


# generate parameters combinations
def parameter_product(parameters):
    params_values = [parameters[key] if isinstance(parameters[key], list) else [parameters[key]] for key in parameters.keys()]
    return [dict(zip(parameters.keys(), combination)) for combination in itertools.product(*params_values)]


# In[ ]:


# training loop
def training_loop(experiment, model_class, parameters, X_train_scaled, y_train, X_test_scaled, y_test):
    runs_parameters = parameter_product(parameters)
    model_params = {}
    
    for i, run_parameters in enumerate(runs_parameters):
#         print(f"Run {i}: {run_parameters}")

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"Run {i}", experiment_id=experiment.experiment_id): # , experiment_id=experiment_id

            model = model_class(**run_parameters)
            model.fit(X_train_scaled, y_train)

            score = mean_squared_error(y_test, model.predict(X_test_scaled), squared=False)
#             print("RMSE score: {:.4f}".format(score))
#             score_cv = rmsle_cv_score(model, X_train_scaled, y_train)
#             print("Cross-Validation RMSE score: {:.4f} (std = {:.4f})".format(score_cv.mean(), score_cv.std()))
            r2 = r2_score(y_test, model.predict(X_test_scaled)) # NEW
#             print("R2-score: {:.4f}".format(r2))

            # generate charts
        #     model_feature_importance(model)
        #     plt.close()
        #     model_permutation_importance(model)
        #     plt.close()
        #     model_tree_visualization(model)

            # get model signature
            signature = infer_signature(model_input=X_train_scaled, model_output=model.predict(X_train_scaled))
            # mlflow: log metrics
            metrics = {
                'RMSE': score,
#                 'RMSE_CV': score_cv.mean(),
                'R2': r2
            }
            log_mlflow_run(model, signature, parameters, metrics)

#         print("")
        
        model_params[f"Run {i}"] = {
            'model': model,
            'params': run_parameters,
            'RMSE': score,
#             'RMSE_CV': score_cv,
            'R2': r2
        }

    return model_params


# In[ ]:


def experiment_initialization(experiment_name):
    # Initialize MLflow experiment

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    return experiment

    # experiment_name = "poland_apartments"
    # experiment_id = mlflow.create_experiment(experiment_name)
    # mlflow.set_experiment(experiment_name)

    # delete default experiment if exits
    # if (mlflow.get_experiment_by_name("Default").lifecycle_stage == 'active'):
    #     mlflow.delete_experiment("0")

    # create model_artifacts directory 
    # !mkdir -p "model_artifacts"


# # 3. For predicting by input data with the best model

# In[ ]:


def check_city_district_radius_floor_rooms(city, district, radius, floor, rooms):
    
    price_df4 = pd.read_csv("poland_apartments_completed.csv")
    floor_values = list(set(price_df4['floor']))
    rooms_values = list(set(price_df4['rooms']))
    
    geo_df_for_check = price_df4.groupby(['city', 'district']).agg({'radius':['min','max']})
    
    if (city, district) not in geo_df_for_check.index:
        print("Invalid city and/or district!")
        return False
    else:
        min_radius = geo_df_for_check.loc[(city, district)][('radius', 'min')]
        max_radius = geo_df_for_check.loc[(city, district)][('radius', 'max')]
        if not (radius >= min_radius and radius <= max_radius):
            print("The radius must be between {0} and {1}".format(round(min_radius, 2), round(max_radius, 2)))
            return False
        elif floor not in floor_values:
            print("The floor must be integer and between {0} and {1}".format(floor_values.min(), floor_values.max()))
            return False
        elif rooms not in rooms_values:
            print("The rooms must be integer and between {0} and {1}".format(rooms_values.min(), rooms_values.max()))
            return False
        else:
            return True
#         else:    
#             for col in [floor, rooms]: # city, district, 
#                 if col not in list(set(price_df4[namestr(col, globals())[0]])):
#                     print(f"Invalid {col}")
#                     return False
    


# In[ ]:


def check_sq(sq):
    
    if (
        type(sq) in [int, float] and 
        sq >= 20 and
        sq <= 100
    ):
        return True
    else:
        print("The sq should be between {0} and {1}".format(20, 100))
        return False


# In[ ]:


def check_year(year, city):
    
    city_foundations = {
        'Warszawa': 1300,
        'Kraków': 990,
        'Poznañ': 1253
    }
    if (
        type(year) is int and 
        year >= city_foundations[city] and
        year <= date.today().year
    ):
        return True
    else:
        print("The year should be integer and between {0} and {1}".format(city_foundations[city], date.today().year))
        return False
    


# In[ ]:


def input_to_df(city, district, radius, floor, rooms, sq, year):
    
    if (check_city_district_radius_floor_rooms(city, district, radius, floor, rooms)
        and 
        check_sq(sq)
        and 
        check_year(year, city)
       ):
        X_check = pd.DataFrame({
            'city': city,
            'district': district,
            'floor': floor, 
            'rooms': rooms, 
            'sq': sq, 
            'year': year,
            'radius': radius
        }, index=[0])
        return X_check
    else:
        return None


# In[ ]:


def set_regressors():

    all_regressors = {
    #     'linreg': LinearRegression,
    #     'ridge': Ridge,
    #     'lasso': Lasso,
    #     'knn': KNeighborsRegressor,
    #     'tree': DecisionTreeRegressor,
    #     'gbr': GradientBoostingRegressor,
        'xgb': XGBRegressor
    }

    grids = {
        'linreg': {},
        'ridge': {
            "alpha": [0.0001], # list(np.logspace(-4, -1, 4))
            "solver": ["lsqr"] # ["sag", "lsqr"]
        },
        'lasso': {
            "alpha": [0.0001], # list(np.logspace(-4, 2, 7))
            "max_iter": [1000]
        },
        'knn': {
            "n_neighbors": [10], # [8, 9, 10]
            "leaf_size": [30], # [25, 30, 35]
            "p": [2],
            "algorithm": ['auto']
        },
        'tree': {
            "min_samples_split": [15], # [10, 15, 20, 25]
            "max_depth": [8], # [6, 8]
        #     "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            "splitter": ['best'] # , 'random'
        },
        'gbr': {
            "learning_rate": [0.2], # [0.2, 0.1, 0.05]
            "max_depth": [5, 6, 7] # [5, 6, 7, 8]
        },
        'xgb': {
            "max_depth": [8], # [6, 7, 8, 9]
            "n_estimators": [200, 300],
            "learning_rate": [0.2] # [0.2, 0.1, 0.05]
        }
    }

    return all_regressors, grids


# In[ ]:


def select_best_model(experiment, all_regressors, grids, X_train_scaled, y_train, X_test_scaled, y_test):
    
    full_model_params = {}
    for reg, model_class in all_regressors.items():
        full_model_params[reg] = training_loop(experiment, model_class, grids[reg], X_train_scaled, y_train, X_test_scaled, y_test)
    
    best_run_df = mlflow.search_runs(order_by=['metrics.R2 DESC'], max_results=1) 
    if len(best_run_df.index) == 0:
        raise Exception(f"Found no runs for experiment '{experiment_name}'")
    
    best_run = mlflow.get_run(best_run_df.at[0, 'run_id'])
    best_model_uri = f"{best_run.info.artifact_uri}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    
    return best_model


# In[ ]:


def predict_by_input(X_check, cat_features, st_scaler, labels_dict, model, X_test_scaled, y_test):
    
    for col in cat_features:
        X_check[col] = X_check[col].apply(lambda x: labels_dict[col][x])

    X_check_scaled = st_scaler.transform(X_check)
    y_check_pred_model = model.predict(X_check_scaled)
    
    score = model.score(X_test_scaled, y_test)
    price_pred = np.round(y_check_pred_model[0])
    print("With a probability of {0}%, the price will be about {1:,.0f} PLN ".format(
        round(score * 100, 1),
        round(price_pred)
    ))
    
    return price_pred, score


# In[3]:


# I NEED THE MAIN FUNCTION WITH FOLLOWING ARGMENTS:
# (BECAUSE THESE WOULD BE INPUT FROM HTML FORM)
    
def main_predicting(city, district, radius, floor, rooms, sq, year):
    
    X_check = input_to_df(city='Warszawa', district='Śródmieście', radius=2, floor=3, rooms=2, sq=40, year=2000)
    price_df4, cat_features, num_features, target = load_data()
    experiment = experiment_initialization("poland_apartments")
    all_regressors, grids = set_regressors()
    
    # run tracking UI in the background
#     get_ipython().system_raw("mlflow ui --port 5000 &")
    
    X_train_scaled, X_test_scaled, y_train, y_test, st_scaler, labels_dict = to_split_and_scale(price_df4, cat_features, num_features, target)
    best_model = select_best_model(experiment, all_regressors, grids, X_train_scaled, y_train, X_test_scaled, y_test)
    price_pred, score = predict_by_input(X_check, cat_features, st_scaler, labels_dict, best_model, X_test_scaled, y_test)
    
    return "With a probability of {0}%, the price will be about {1:,.0f} PLN ".format(round(score * 100, 1), round(price_pred))


# In[ ]:





# In[ ]:





# In[ ]:


# def predicting_by_experiment(experiment, all_regressors, grids, X_train_scaled, y_train, X_test_scaled, y_test, cat_features, st_scaler, labels_dict):
    
#     full_model_params = {}
    
#     for reg, model_class in all_regressors.items():
# #         print(f"{reg}:".upper())
#         full_model_params[reg] = training_loop(experiment, model_class, grids[reg], X_train_scaled, y_train, X_test_scaled, y_test)
# #         print()
        
#     best_run_df = mlflow.search_runs(order_by=['metrics.R2 DESC'], max_results=1) 
#     if len(best_run_df.index) == 0:
#         raise Exception(f"Found no runs for experiment '{experiment_name}'")

#     best_run = mlflow.get_run(best_run_df.at[0, 'run_id'])
#     best_model_uri = f"{best_run.info.artifact_uri}/model"
#     best_model = mlflow.sklearn.load_model(best_model_uri)

# #     print(f"Run parameters: {best_run.data.tags['estimator_name']}")
# #     print(f"Run parameters: {best_run.data.params}")
# #     print("Run score: R2 = {:.4f}".format(best_run.data.metrics['R2']))
    
# #     model_name = best_run.data.tags['estimator_name']    
# #     best_grid2 = {k: float(v) if '.' in v else int(v) for k, v in best_run.data.params.items()}
# #     best_model2 = globals()[model_name](**best_grid2)
# #     best_model2.fit(X_train_scaled, y_train)
    
#     X_check = input_to_df(city='Warszawa', district='Śródmieście', radius=2, floor=3, rooms=2, sq=40, year=2000)
    
#     price_pred, score = predict_by_input(X_check, cat_features, st_scaler, labels_dict, best_model, X_test_scaled, y_test)
    
#     return price_pred, score


# In[ ]:


# def model_feature_importance(model):
#     feature_importance = pd.DataFrame(
#         model.feature_importances_,
#         index=X_train_scaled, # X_train_scaled.columns
#         columns=["Importance"],
#     )

#     # sort by importance
#     feature_importance.sort_values(by="Importance", ascending=False, inplace=True)

#     # plot
#     plt.figure(figsize=(12, 8))
#     sns.barplot(
#         data=feature_importance.reset_index(),
#         y="index",
#         x="Importance",
#     ).set_title("Feature Importance")
#     # save image
#     plt.savefig("model_artifacts/feature_importance.png", bbox_inches='tight')


# In[ ]:


# def model_permutation_importance(model):
#     p_importance = permutation_importance(model, X_test_scaled, y_test, random_state=42, n_jobs=-1)

#     # sort by importance
#     sorted_idx = p_importance.importances_mean.argsort()[::-1]
#     p_importance = pd.DataFrame(
#         data=p_importance.importances[sorted_idx].T,
#         columns=X_train.columns[sorted_idx]
#     )

#     # plot
#     plt.figure(figsize=(12, 8))
#     sns.barplot(
#         data=p_importance,
#         orient="h"
#     ).set_title("Permutation Importance")

#     # save image
#     plt.savefig("model_artifacts/permutation_importance.png", bbox_inches="tight")

