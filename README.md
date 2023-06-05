# Price Predicting for Apartments in Poland

Predicting the price of apartment by its features (square, floor, district, distance from the center of the city, number of rooms, year of construction).

Libraries: numpy, pandas, opendatasets, geopandas, geopy, sklearn, xgboost, mlxtend, seaborn, matplolib, time, itertools, mlflow, fastapi

[![Code Climate][codeclimate-badge]][codeclimate-link]

[codeclimate-badge]: https://codeclimate.com/github/am-tropin/poland-apartment-prices.svg
[codeclimate-link]: https://codeclimate.com/github/am-tropin/poland-apartment-prices



## Table of contents
- [Datasets](#datasets)
- [Machine learning problem](#machine-learning-problem)
- [Run locally](#run-locally)
- [MLflow Tracking](#mlflow-tracking)


## Datasets

- The dataset of prices is from [Kaggle](https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland). 

- The dataset of cities' disctricts were downloaded from here: [Warsaw](https://github.com/andilabs/warszawa-dzielnice-geojson), [Krakow](https://github.com/andilabs/krakow-dzielnice-geojson), [Poznan](https://sip.poznan.pl/sip/dzielnice/get_dzielnice).

- The final dataset for ML problem is created in [data_processing.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/data_processing.ipynb) file and saved as [poland_apartments_completed.csv](https://github.com/am-tropin/poland-apartment-prices/blob/main/poland_apartments_completed.csv) file.


## Machine learning problem

- The given problem was solved by using **XGBoost Regressor**. It shows the best **R2-score (87%)** in comparison with other models: Linear Regression, Ridge, Lasso, Bagging Regressor, AdaBoost (by 65%), Decision Tree (77%), k-Nearest Neighbors (78%), Random Forest, Stacked Ensembles (by 83%) and Gradient Boosting (85%). Besides, the predictor of price by custom data (using XGBoost model) was built. The result is in the [model_evaluation.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/model_evaluation.ipynb) file.


The same code for both data processing and model evaluation also contains in the [Poland_apartments__full.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/Poland_apartments__full.ipynb) file.



## Run locally

1. Clone the project:
```bash
  git clone https://github.com/am-tropin/poland-apartment-prices
```

2. Go to the project directory:
```bash
  cd poland-apartment-prices
```

<!-- Create vitual enviroment and install dependencies
```bash
  virtualenv venv
  source venv/bin/activate
  pip install -r requirements.txt
``` -->

3. Start the server:
```bash
  uvicorn main:app --reload
```

4. Go to web-browser 
```bash
  http://127.0.0.1:8000/docs/
```
and use the following box:

- **Get Main Predicting**: Type city and district names, distance from the center of the city, floor number and number of rooms, apartment square and year of construction.

Or 

5. Go to web-browser and use one the following types of links to get the same info in clear dictionary view:

- 5.1.
```bash
  http://127.0.0.1:8000/price_predictor/Warszawa_Śródmieście_2_3_2_40_2000
```



## MLflow Tracking

1. Start the MLflow UI:
```bash
  mlflow ui
```

<!-- 2. Go to the project directory:
```bash
  mlflow ui --backend-store-uri /Users/user/Documents/GitHub/poland-apartment-prices/mlruns
``` -->

2. Go to web-browser
```bash
  http://127.0.0.1:5000/
```
and choose **poland_apartments** experiment for comparing the models.



