# Price Predicting for Poland Apartments

Predicting the price of apartment by its features (square, floor, district, distance from the center of the city, number of rooms, year of construction)

Libraries: numpy, pandas, geopandas, sklearn, seaborn, matplolib


## Table of contents
- [Datasets](#datasets)
- [Machine learning approach](#machine-learning-approach)


## Datasets

- The dataset of prices is from [Kaggle](https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland). 

- The dataset of cities' disctricts was downloaded here: [Warsaw](https://github.com/andilabs/warszawa-dzielnice-geojson), [Krakow](https://github.com/andilabs/krakow-dzielnice-geojson), [Poznan](https://sip.poznan.pl/sip/dzielnice/get_dzielnice).


## Machine learning approach

- The given problem was solved by using **the scikit-learn RandomForestRegressor**. The result is in the [Poland_apartments_main.ipynb](https://github.com/am-tropin/poland-apartment-prices/blob/main/Poland_apartments_main.ipynb) file.
