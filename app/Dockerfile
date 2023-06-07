FROM python:3.11-slim
# jupyter/scipy-notebook 
# python:3.7-slim 

# Copy requirements first to validate pip cache to prevent rebuild
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Create the environment:
RUN apt update
RUN apt install git -y
# RUN apt-get install gdal-bin
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the other files
COPY . /app

# Run jupyter
# RUN python _loaders.py

CMD ["python", "main.py"]
