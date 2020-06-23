import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    function load_data
    input:
    file path for messages (string)
    file path for categories (string)
    output:
    concanated pandas DataFrame of messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
    
    return df

def clean_data(df):
    '''
    function clean_data
    split categories into separate category columns
    Convert category values to just numbers 0 or 1
    categories column in df with new category columns
    Remove duplicates
    input: a concanated DataFrame 
    output: cleaned DataFrame
    '''
     # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat = ';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x[:-2]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        categories[column].replace ('2', '1', inplace = True)
         # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    df2 = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df2.drop_duplicates(subset = 'message', inplace = True)
    
    return df2

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False)


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