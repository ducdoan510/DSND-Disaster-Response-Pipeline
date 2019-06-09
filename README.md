# Disaster Response Pipeline Project

### Overview

This project analyzes disaster data from [Figure Eight](https://www.figure-eight.com) to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline will be created to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Structure

1. Flask web app: _app_ folder contains a _run.py_ that can be run to start the Flask web server
2. ETL Pipleline: _data_ folder contains 2 csv files used in this project and a _process_data.py_ including ETL steps
3. Machine Learning Pipeline: _model_ folder contains _train_classifier.py_ file to build a model on the given dataset

### Acknowledge

This project is a part of Udacity Data Science Nanodegree program. The dataset is obtained from [Figure Eight](https://www.figure-eight.com)
