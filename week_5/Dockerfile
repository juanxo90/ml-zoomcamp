# creating the python
FROM python:3.7.13

# installing libs
RUN pip install pipenv

# creating working dir
WORKDIR /app

# Coy the files to get the pipenv the ./ to the current directory
COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

# get the scripts used
COPY ["predict.py", "model_C=1.0.bin", "./"]

# Expose the port
EXPOSE 9696

# Entry point run the python gunicorn command 
# to get the port of the flask app and run the predicts!
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]