#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pytest


# In[10]:


from datetime import datetime
import pandas as pd


# In[1]:


import sys
sys.path.append('../app/')

from functions.function_store import load_data # depend on "poland_apartments_completed.csv"
from functions.function_store import namestr, parameter_product, check_sq, check_year # don't depend on any functions

# 1. For model evaluation
from functions.function_store import to_split_and_scale, reg_model_implementation, show_feature_importances # don't depend on any functions

# 2. For MLFlow tracking
from functions.function_store import rmsle_cv_score, log_mlflow_run, experiment_initialization # don't depend on any functions
from functions.function_store import training_loop # depend on other functions

# 3. For predicting by input data with the best model
from functions.function_store import check_city_district_radius_floor_rooms # depend on "poland_apartments_completed.csv"
from functions.function_store import input_to_df # depend on other functions
from functions.function_store import select_best_model, main_predicting # depend on other functions
from functions.function_store import set_regressors, predict_by_input # don't depend on any functions


# In[ ]:





# In[ ]:


def test_for_1_check_sq():
    assert check_sq(20) is True

def test_for_2_check_sq():
    assert check_sq(99.9) is True

def test_for_3_check_sq():
    assert check_sq(19) is False

def test_for_4_check_sq():
    assert check_sq(100.1) is False


# In[ ]:

def test_for_11_check_year():
    assert check_year(2020, 'Warszawa') is True

def test_for_12_check_year():
    assert check_year(920, 'Warszawa') is False

def test_for_13_check_year():
    assert check_year(2130, 'Warszawa') is False

def test_for_21_check_year():
    assert check_year(2020, 'Kraków') is True

def test_for_22_check_year():
    assert check_year(920, 'Kraków') is False

def test_for_23_check_year():
    assert check_year(2130, 'Kraków') is False

def test_for_31_check_year():
    assert check_year(2020, 'Poznañ') is True

def test_for_32_check_year():
    assert check_year(920, 'Poznañ') is False

def test_for_33_check_year():
    assert check_year(2130, 'Poznañ') is False





# In[ ]:





# In[9]:


# def test_for_corr_stod():
#     assert stod("2023-05-30") == datetime(year=2023, month=5, day=30).date()
    
# def test_for_incorr_stod():
#     with pytest.raises(ValueError, match="Incorrect data format, should be YYYY-MM-DD"):
#         stod("202-05-30")


# In[17]:


# def test_for_corr_daterange():
#     assert list(daterange(datetime(year=2023, month=5, day=30).date(), datetime(year=2023, month=6, day=1).date())) == [
#         datetime(year=2023, month=5, day=30).date(), 
#         datetime(year=2023, month=5, day=31).date(), 
#         datetime(year=2023, month=6, day=1).date()]


# In[ ]:


# def test_for_1_check_roundness():
#     assert check_roundness(2000, 2, 2) is True

# def test_for_2_check_roundness():
#     assert check_roundness(1200, 2, 2) is False

# def test_for_valuerror_check_roundness():
#     with pytest.raises(ValueError, match="Incorrect data format of importance, should be in 1, 2 or 3."):
#         check_roundness(500, 0, 2)


# In[ ]:


# def test_for_past_rule_days_divisibility():
#     assert rule_days_divisibility(datetime(year=2023, month=5, day=30).date(), datetime(year=2022, month=7, day=31).date(), 2) is None


# In[2]:


# def test_for_notempty_date_dict_to_df():
#     dict_1 = {'date': datetime(year=2023, month=5, day=29).date(),
#               'event': 'event 1', 
#               'amount': 5, 
#               'unit': 'day'}
#     df_1 = pd.DataFrame.from_dict(dict_1, orient='index').transpose()
#     dict_2 = {'date': "2023-05-31",
#                  'event': 'event 3', 
#                  'amount': 32, 
#                  'unit': 'day'}
#     df_2 = pd.DataFrame.from_dict(dict_2, orient='index').transpose()
#     assert pd.testing.assert_frame_equal(date_dict_to_df(df_1, datetime(year=2023, month=5, day=31).date(), 'event 3', 32, 'day'), pd.concat([df_1, df_2]).reset_index(drop=True)) is None

    
# def test_for_empty_date_dict_to_df():
#     null_df = pd.DataFrame(columns=['date', 'event', 'amount', 'unit'])
#     dict_2 = {'date': "2023-05-31",
#                  'event': 'event 3', 
#                  'amount': 32, 
#                  'unit': 'day'}
#     df_2 = pd.DataFrame.from_dict(dict_2, orient='index').transpose()
#     assert pd.testing.assert_frame_equal(date_dict_to_df(null_df, datetime(year=2023, month=5, day=31).date(), 'event 3', 32, 'day'), df_2) is None

    


# In[ ]:





# In[ ]:





# In[ ]:




