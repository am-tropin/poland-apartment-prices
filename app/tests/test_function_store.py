#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pytest


# In[11]:


import pandas as pd


# In[15]:


import sys
sys.path.append('../app/')

# from functions.function_store import load_data # depend on "poland_apartments_completed.csv"
from functions.function_store import namestr, parameter_product, check_sq, check_year # don't depend on any functions

# 1. For model evaluation
from functions.function_store import to_split_and_scale, reg_model_implementation, show_feature_importances # don't depend on any functions

# 2. For MLFlow tracking
from functions.function_store import rmsle_cv_score, log_mlflow_run, experiment_initialization # don't depend on any functions
from functions.function_store import training_loop # depend on other functions

# 3. For predicting by input data with the best model
from functions.function_store import check_city_district_radius_floor_rooms, input_to_df # depend on other functions
from functions.function_store import select_best_model, main_predicting # depend on other functions
from functions.function_store import set_regressors, predict_by_input # don't depend on any functions


# In[19]:


from datatest import working_directory

@pytest.fixture(scope="session") #(scope='module')
@working_directory(__file__)
def grab_test_file():
    file_path = 'test_file.csv'
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        pytest.skip(f'cannot find {file_path}')

# !!! https://datatest.readthedocs.io/en/stable/reference/data-handling.html
# https://github.com/shawnbrown/datatest/blob/bf136eab23c2b6ea36c201e1446fca9243c3fba6/docs/discussion/tips-and-tricks.rst?plain=1#L448


# In[ ]:


def test_1_check_city_district_radius_floor_rooms():
#     assert check_city_district_radius_floor_rooms(TEST_APARTMENTS_DF, "Kraków", "Śródmieście", 2, 3, 2) is False
    assert check_city_district_radius_floor_rooms(grab_test_file, "Kraków", "Śródmieście", 2, 3, 2) is False
    
    
#     assert check_city_district_radius_floor_rooms("Warszawa", "Śródmieście", 2, 3, 2)


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





# In[ ]:





# In[ ]:





# In[14]:


# import os 
# os.listdir()


# In[ ]:





# In[16]:


# sys.path.append('../app/tests/')
# TEST_APARTMENTS_FILE = "../app/tests/test_file1.csv"
# TEST_APARTMENTS_DF = pd.read_csv(TEST_APARTMENTS_FILE)

# test_df = pd.read_csv('original_file.csv')
# # test_df.shape

# test1_df = test_df.groupby('city').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)
# # test1_df.shape
# test1_df.to_csv("test_file.csv")

# https://stackoverflow.com/questions/22472213/python-random-selection-per-group


# In[17]:


# test_df = pd.read_csv('test_file.csv')
# test_df.shape


# In[ ]:





# In[ ]:


# TEST_APARTMENTS_FILE = "test_file.csv"

# # Creating a fixture that grabs a file before running a test
# @pytest.fixture(scope="session") #(scope="module")
# def grab_test_file():
#     return pd.read_csv(TEST_APARTMENTS_FILE)

# # Using the fixture in a test function
# def test_file_content(grab_test_file):
#     assert grab_test_file.shape == (10, 5)

# https://towardsdatascience.com/writing-better-code-through-testing-f3150abec6ca


# In[ ]:


# def test_1_load_data(grab_test_file):
#     assert grab_test_file.shape == (4315, 20)


# In[ ]:


# def test_01_load_data():
# #     assert load_data(TEST_APARTMENTS_DF)[0].shape == (4315, 20)
#     assert load_data(grab_test_file)[0].shape == (4315, 20)

# def test_11_load_data():
# #     assert load_data(TEST_APARTMENTS_DF)[1] == ['city', 'district']
#     assert load_data(grab_test_file)[1] == ['city', 'district']

# def test_21_load_data():
# #     assert load_data(TEST_APARTMENTS_DF)[2] == ['floor', 'rooms', 'sq', 'year', 'radius']
#     assert load_data(grab_test_file)[2] == ['floor', 'rooms', 'sq', 'year', 'radius']
    
# def test_31_load_data():
# #     assert load_data(TEST_APARTMENTS_DF)[3] == ['price']
#     assert load_data(grab_test_file)[3] == ['price']


# In[ ]:


# def check_city_district_radius_floor_rooms(city, district, radius, floor, rooms):
    
#     price_df4 = load_data()[0]
#     floor_values = list(set(price_df4['floor']))
#     rooms_values = list(set(price_df4['rooms']))
    
#     geo_df_for_check = price_df4.groupby(['city', 'district']).agg({'radius':['min','max']})
    
#     if (city, district) not in geo_df_for_check.index:
#         print("Invalid city and/or district!")
#         return False
#     else:
#         min_radius = geo_df_for_check.loc[(city, district)][('radius', 'min')]
#         max_radius = geo_df_for_check.loc[(city, district)][('radius', 'max')]
#         if not (radius >= min_radius and radius <= max_radius):
#             print("The radius must be between {0} and {1}".format(round(min_radius, 2), round(max_radius, 2)))
#             return False
#         elif floor not in floor_values:
#             print("The floor must be integer and between {0} and {1}".format(floor_values.min(), floor_values.max()))
#             return False
#         elif rooms not in rooms_values:
#             print("The rooms must be integer and between {0} and {1}".format(rooms_values.min(), rooms_values.max()))
#             return False
#         else:
#             return True


# In[ ]:





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




