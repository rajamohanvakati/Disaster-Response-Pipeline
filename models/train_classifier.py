import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import time


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier 
from sklearn.naive_bayes import MultinomialNB

import pickle



def load_data(database_filepath):
    '''
    Funtion loads a dataset from database made by process_data.py
    
    Input:
    1. database_filepath: path for a database file
    
    Output:
    1. X: dataframe having messages to be used for machine learning
    2. Y: dataframe having categories to be used for machine learning as targets
    3. category_names: list of category names from the Y dataframe
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    
    # split X and Y from df
    X = df['message'].values
    Y = df.iloc[:,4:]
    
    # create category_names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names
   

def tokenize(text):
    '''
    Function splits text into separate words and gets a word lowercased and removes whitespaces at the ends of a word. The funtions also cleans irrelevant stopwords.
    
    Input:
    1. text: text message
    
    Output:
    1. Clean_tokens : list of tokenized clean words
    '''
    # Get rid of other sepcial characters   
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)
    
    # Remove stop words    
    stopwords = nltk.corpus.stopwords.words('english')
    clean_tokens = [token for token in clean_tokens if token not in stopwords]
    
    return clean_tokens
      
class TokenCounter(BaseEstimator, TransformerMixin):
    '''
    Customized class for counting the number of tokens
    '''
    def count_tokens(self, text):
        '''
        Funtion counts the number of tokens in a sentence
        Input:
        1. text: a one sentence message
        Output:
        1. number_of_tokens: number of tokens 
        '''
        tokens = nltk.word_tokenize(text)
        number_of_tokens = len(tokens)
        return  number_of_tokens
    
    def fit(self, X, y=None):
        '''
        Function for fitting
        Input :
        1. X
        Output:
        1. X
        '''
        return self

    def transform(self, X):
        '''
        Function for transforming a dataset. It uses count_tokens function to count the number of tokens and creates a dataframe having the result of the each counting.
        Input:
        1. X
        Output:
        1. pd.DataFrame(X_tagged): dataframe having the number of tokens from each sentence
        '''
        X_tagged = pd.Series(X).apply(self.count_tokens)
        return pd.DataFrame(X_tagged)

def build_model(X_train, Y_train):
    '''
    Funtion builds first a machine learning pipeline using FeatureUnion and Pipeline. Then the best parameters are searched by GridSearchCV. Lastly, a model with the best parameters is returned.
    Input:
    none
    Output:
    1. model: 
    '''
    # Build a machine learning pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('counttk', TokenCounter())
    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    parameters = {
    'features__text_pipeline__vect__max_df': (0.5,1.0),
    'clf__estimator__n_estimators': [50,100],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'counttk': 0.5},
            {'text_pipeline': 0.5, 'counttk': 1})
    }
    
    # Use grid search to find better parameters
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=10)
    cv.fit(X_train, Y_train)
    
    # Update the parameters of model using the result from GridSearch
    model = pipeline
    model.set_params(**cv.best_params_)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function for evaluation of the model by build_model(). It prints out classification_report of each category name
    Input:
    1. model: model built by build_model
    2. X_test: test dataset made by main()
    3. Y_test: real category values made by main()
    4. Category_names : name of categories
    '''
    # Save a result of predition
    Y_pred = model.predict(X_test)
    
    # Show classification_report
    for i in range(0, len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[category_names[i]], Y_pred[:, i]))
        print("f1_score of {} is {:.2f}".format(category_names[i], f1_score(Y_test[category_names[i]], Y_pred[:, i])))
        print("**"*30)
    


def save_model(model, model_filepath):
    '''
    Function saves a completed model
    Input:
    1. model: trained model to save
    2. model_filepath: path for saving a trained model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()