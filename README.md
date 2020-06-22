# Disaster Response Pipeline Project
## Created by Zixing (Cecile) Wang as part of fulfillment for Udacity Data Science Nanodegree

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
