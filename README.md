# Price Predicting for Apartments in Poland

Predicting the price of apartment by its features (square, floor, district, distance from the center of the city, number of rooms, year of construction).

Libraries: numpy, pandas, opendatasets, geopandas, geopy, sklearn, xgboost, mlxtend, seaborn, matplolib, time, itertools, mlflow, fastapi

[![codecov][codecov-badge]][codecov-link] [![Code Climate][codeclimate-badge]][codeclimate-link]

[codecov-badge]: https://codecov.io/gh/am-tropin/poland-apartment-prices/coverage.svg
[codecov-link]: https://codecov.io/gh/am-tropin/poland-apartment-prices

[codeclimate-badge]: https://codeclimate.com/github/am-tropin/poland-apartment-prices.svg
[codeclimate-link]: https://codeclimate.com/github/am-tropin/poland-apartment-prices



## Table of contents
- [Datasets](#datasets)
- [Machine learning problem](#machine-learning-problem)
- [Run locally](#run-locally)
- [MLflow Tracking](#mlflow-tracking)
- [Docker](#docker)


## Datasets

- The dataset of prices is from [Kaggle](https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland). 

- The dataset of cities' disctricts were downloaded from here: [Warsaw](https://github.com/andilabs/warszawa-dzielnice-geojson), [Krakow](https://github.com/andilabs/krakow-dzielnice-geojson), [Poznan](https://sip.poznan.pl/sip/dzielnice/get_dzielnice).

- The final dataset for ML problem is created in [data_processing.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/app/data_processing.ipynb) file and saved as [poland_apartments_completed.csv](https://github.com/am-tropin/poland-apartment-prices/blob/main/app/poland_apartments_completed.csv) file.


## Machine learning problem

- The given problem was solved by using **XGBoost Regressor**. It shows the lowest **MAPE (Mean absolute percentage error) = 9%** in comparison with other models: Linear Regression, Ridge, Lasso, Bagging Regressor (by 26-28%), Decision Tree (17%), k-Nearest Neighbors, Random Forest, Stacked Ensembles (by 14%), Gradient Boosting (13%), AdaBoost for XGBoost (9%). Besides, the predictor of price by custom data (using XGBoost model) was built. The result is in the [model_evaluation.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/app/model_evaluation.ipynb) file.


The same code for both data processing and model evaluation also contains in the [Poland_apartments__full.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/Poland_apartments__full.ipynb) file.



## Run locally

1. Clone the project:
```bash
  git clone https://github.com/am-tropin/poland-apartment-prices
```

2. Go to the project directory:
```bash
  cd poland-apartment-prices/app
```

<!-- Create vitual enviroment and install dependencies
```bash
  virtualenv venv
  source venv/bin/activate
  pip install -r requirements.txt
``` -->

3. Start the server:
```bash
  uvicorn app.main:app --reload
```

4. Go to web-browser 
```bash
  http://127.0.0.1:8000/docs/
```
and use the following box: **Get Main Predicting**. Type city and district names, distance from the center of the city, floor number and number of rooms, apartment square and year of construction.

Or 

5. Go to web-browser and use the following link to get the same info after typing the parameters:

```bash
  http://127.0.0.1:8000/price/_
```

Or 

6. Go to web-browser and use the following type of links to get the same info in clear dictionary view:

```bash
  http://127.0.0.1:8000/main_predicting/Warszawa_Śródmieście_2_3_2_40_2000
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


## Docker

1. Clone the project:
```bash
  git clone https://github.com/am-tropin/poland-apartment-prices
```

2. Go to the project directory:
```bash
  cd poland-apartment-prices/app
```

3. Create a docker container:
```bash
  docker build -t price-predictor .
```


