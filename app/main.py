#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import sys
# sys.path.append('../')
from functions.function_store import main_predicting

# API
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates


# In[3]:


app = FastAPI()
templates = Jinja2Templates(directory="templates/")


@app.get("/")
async def root():
    return "Welcome to the Price Predicting for Apartments in Poland!"


# without html

@app.get("/main_predicting/{city}_{district}_{radius}_{floor}_{rooms}_{sq}_{year}")
async def get_main_predicting(city: str, district: str, radius: float, floor: int, rooms: int, sq: float, year: int):
    return main_predicting(city, district, radius, floor, rooms, sq, year)


# for html

@app.get("/price_html/{city}_{district}_{radius}_{floor}_{rooms}_{sq}_{year}")
async def get_main_predicting_html(city: str, district: str, radius: float, floor: int, rooms: int, sq: float, year: int):
    return {"Price prediction:": main_predicting(city, district, radius, floor, rooms, sq, year)}

@app.get("/price/{form}")
def form_post_price(request: Request):
    result = "Write apartment's parameters"
    return templates.TemplateResponse('form_predictor.html', context={'request': request, 'result': result})

@app.post("/price/{form}")
def form_post_price(request: Request, city: str = Form(...), district: str = Form(...), radius: float = Form(...), floor: int = Form(...), rooms: int = Form(...), sq: float = Form(...), year: int = Form(...)):
    result = main_predicting(city, district, radius, floor, rooms, sq, year)
    return templates.TemplateResponse('form_predictor.html', context={'request': request, 'result': result}) # .to_html()


# In[ ]:





# In[ ]:





# In[2]:


# input_dict = {
#     'city': 'Warszawa',
#     'district': 'Śródmieście',
#     'floor': 3, 
#     'rooms': 2, 
#     'sq': 40, 
#     'year': 2000,
#     'radius': 2
# }

# main_predicting(**input_dict)


# In[ ]:





# In[ ]:





# In[ ]:




