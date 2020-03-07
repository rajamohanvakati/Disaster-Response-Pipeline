"""
PREPROCESSING DATA
Disaster Response Pipeline Project
Sample Script Execution:
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function loads two datasets to make corresponding dataframes and merge into one dataframe.
    
    Input:
    1. messages_filepath: path for a messages csv file
    2. categories_filepath: path for a categories csv file
    
    Output:
    1. df: pandas dataframe after merging of two inputs
    '''
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge dataset
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    '''
    Funtion cleans data in the df dataframe after loading by the load_data funtion. It splits categories into separate category columns, converts category values to binary data (0 or 1), and remove duplicates. 
    
    Input:
    1. df: dataframe made by load_data function
    
    Output:
    1. df: cleaned dataframe 
    '''
    # split a column into 36 individual category columns
    categories = df.categories.str.split(pat=";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to have values only 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # set values over 1 to 1
        categories.loc[categories[column] > 1, column] = 1
        
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # first check the number of duplicates and drop the duplicates
    duplicate_rowdf = df[df.duplicated()]
    df_cleaned_1 = df.drop(labels=duplicate_rowdf.index, axis=0)
    # second check the number of duplicates drop the duplicates
    duplicate_rowdf_id = df_cleaned_1[df_cleaned_1.duplicated('id')]
    df_cleaned_2 = df_cleaned_1.drop(labels=duplicate_rowdf_id.index, axis=0)

    # third check the number of duplicates drop the duplicates
    duplicate_rowdf_message = df_cleaned_2[df_cleaned_2.duplicated('message')]
    df_cleaned_3 = df_cleaned_2.drop(labels=duplicate_rowdf_message.index, axis=0)

    df = df_cleaned_3
    return df
        
def save_data(df, database_filename):
    '''
    Funtion saves a cleaned dataframe to sqlite database
    
    Input:
    1. df: a cleaned dataframe after clean_data funtion
    2. database_filename: a designated filename of database
    '''
    
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists ='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()