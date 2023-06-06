#!/usr/bin/env python
# coding: utf-8

# In[1]:


from functions.functions import main_predicting

# API
from fastapi import FastAPI, Request, Form
# from fastapi.templating import Jinja2Templates


# In[ ]:


app = FastAPI()
# templates = Jinja2Templates(directory="templates/")


@app.get("/")
async def root():
    return "Welcome to the Price Predicting for Apartments in Poland!"


# without html

@app.get("/price_predictor/{date}")
async def get_main_predicting(city: str, district: str, radius: float, floor: int, rooms: int, sq: float, year: int):
    return main_predicting(city, district, radius, floor, rooms, sq, year)


# for range

# @app.get("/price_html/{date1}_{date2}")
# async def get_main_predicting_html(date1: str, date2: str):
#     return {"Anniversaries in the range:": main_predicting(date1, date2)}

# @app.get("/price/{form}")
# def form_post_price(request: Request):
#     result = "Write start and end dates as YYYY-MM-DD"
#     return templates.TemplateResponse('form_range.html', context={'request': request, 'result': result})

# @app.post("/price/{form}")
# def form_post_price(request: Request, date1: str = Form(...), date2: str = Form(...)):
#     result = main_predicting(date1, date2)
#     return templates.TemplateResponse('form_range.html', context={'request': request, 'result': result.to_html()})


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




