import sys
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

import pickle

def load_data(database_filepath):
    """
    function load data
    input:
    file path of cleaned data
    output:
    X and Y as DataFrames for training and testing
    """
    # read in file
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', con = engine)
    X = df.message

    y_names = df.columns.drop(['id', 'message', 'genre', 'original'])
    Y = pd.DataFrame(df[y_names])
    return X, Y, y_names

def tokenize(text):
    """
    a tokenization function to process your text data
    input: 
    text as a string
    output: 
    tokenized and cleaned list
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a model pipeline
    GridSearchCV for optimized parameters.
    Parameters tried: 
    n_estimators; mins_samples_split; ngram_range
    two models were tried out:
    Random Forest Classifier and Multi-nomial NB
    """
    # text processing and model pipeline
    pipeline = Pipeline([
     ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = [
    {"clf": [MultiOutputClassifier(RandomForestClassifier())],
     "clf__estimator__n_estimators": [10, 20],
     "clf__estimator__max_depth":[8],
     "clf__estimator__random_state":[42],
     "clf__estimator__min_samples_split": [2, 3, 4],
     'vect__ngram_range': [(1, 1), (1, 2)]
    },
    {"clf": [MultiOutputClassifier(MultinomialNB())]}
    ]

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(
    estimator = pipeline,
    param_grid = parameters, 
    n_jobs=4
    )

    return cv     

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model
    input:
    model: sciket learn model
    X_test, Y_test: arrays used for test
    category_name: list of categories
    output: 
    print out the f1_score, precision, recall and report for each category
    '''
    Y_pred = model.predict(X_test)
    # display results
    for i, col in enumerate(Y_test):
        print(f'{col} category:', classification_report(np.hstack(np.array(Y_test[col])),np.hstack(Y_pred[:,i])))
    
    

def save_model(model, model_filepath):
    '''
    function saves the model into a pickle file
    '''
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    '''
    Run the machine learning pipeline
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Best model parameters: \n', model.best_estimator_.get_params())
        
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
