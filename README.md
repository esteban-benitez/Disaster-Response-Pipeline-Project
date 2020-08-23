# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### This is a Web App utilizng an ETL, and Machine Learning Pipeline with Python; the following libraries are required in each:

* sys
* pandas
* numpy
* sqlalchemy
* json
* plotly
* flask
* plotly.graph_objs * Bar
* joblib
* sqlalchemy * create_engine
* nltk
* pickle
* re
* sklearn

The dataset is a collection of text messages sent to emergency services, and through an ETL and machine learning pipeline, the messages are classified through a predetermined category set:

    Related, Request, Offer, Aid Related, Medical Help, Medical Products, Search And Rescue, Security, Military, Child Alone, Water, Food, Shelter, Clothing, Money, Missing People, Refugees, Death, Other Aid, Infrastructure Related, Transport, Buildings, Electricity, Tools, Hospitals, Shops, Aid Centers, Other Infrastructure, Weather Related, Floods, Storm, Fire, Earthquake, Cold, Other Weather, Direct Report.

Purpose of the web app, is to disseminate the appropriate resources when large numbers of text messages are received on a network through the following steps.

    1.  ETL Pipeline
        1.  Data is passed in through the disaster_messages.csv file at run time, with an accompany disaster_categories.csv file which provides the resource categories which will be assigned by the Machine Learning Pipeline.
        2.  The data has been transformed with various data cleansing procedures.
        3.  DataFrame is converted and stored in a SQLite3 .db file for processing by the ML Pipeline.

    2.  ML Pipeline
        1.  The SQLite3 engine is used to retrieve the previously created .db file, which is then converted to a Pandas DataFrame object.
        2.  Through SciKit the data is split into a training, and test set, which is fed into the ML Pipeline object consisting of:
            * CountVectorizer
            * TfidfTransformer
            * RandomForestClassifier
        3.  After the pipeline is fitted, the test sets are evaluated, and the classification_report is printed to console.
        4.  Model is saved to file classifier.plk.
   
    3. The Web App is produced with a text input box to classifiy exogenous messages, and also a displacement chart of each categories of the previously uploaded categorical data sets.


