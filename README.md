# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [Instructions](#instructions)

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:
* punkt
* wordnet
* stopwords


## Project Motivation<a name="motivation"></a>

In this project, I appled data engineering, natural language processing, and machine learning skills to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.


## File Descriptions <a name="files"></a>

There are three main foleders:
1. data
- disaster_categories.csv: dataset including all the categories 
- disaster_messages.csv: dataset including all the messages
- process_data.py: ETL pipeline scripts to read, clean, and save data into a database
- DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
- train_classifier.py: machine learning pipeline scripts to train and export a classifier
- classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
- run.py: Flask file to run the web application
- templates contains html file for the web applicatin

## Results<a name="results"></a>

1. An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
2. A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
3. A Flask app was created to show data visualization and classify the message that user enters on the web page.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project. 
## Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
`python run.py`

3. Go to http://0.0.0.0:3001/

