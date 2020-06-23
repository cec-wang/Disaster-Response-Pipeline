# Disaster Response Pipeline Project
## Created by Zixing (Cecile) Wang as part of fulfillment for Udacity Data Science Nanodegree
### Purpose:
The purpose of this project is to automatically categorize a string of text based on the words used. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Note:
1. generate and fetch the pickle file for the model in the same enviroment. Fetch a pickle file generated from a different environment might yield error. classifier.pkl included in the package can be used directly at Udacity workspace. But you need to generate new classifier.pkl if using Jupyter Notebook or any other environment.  

2. The package has been successfully run at workspace. 

3. Last modification date: Jun 22, 2020

### Libraires Used
*run.py* 
* json
* plotly
* pandas
* nltk
* Flask
* sklearn
* SQalchemy
*process_data.py*
* sys
* pandas
* SQAlchemy
*train_classifier*
* sys
* pandas
* SQAlchemy
* nltk
* sklearn
* pickle

### Package Composition
1. Folder *app*
    * Folder *templates*  
        * *go. html*: html template for the webpage with the category output
        * *aster.html*: html template for the main webpage
    * *run.py* : python file that shows the webpage with graphs and prediction
2. Folder *data*
    * *disaster_categories.csv*: message categories raw data
    * *disaster_messages.csv*: messages raw data
    * *DisasterResponse.db*: database generated from process_data.py
    * *process_data.py*: python pipline for data cleaning
3. Folder *models*
    * *train_classifier.py*: python pipeline for model training
    * *classifier.pkl*: model pickle file generated from train_classifier.py which can be run with run.py. However, this package is not the lastest. Stronly encourage you to generate your own pickle file. 
4. README.md
