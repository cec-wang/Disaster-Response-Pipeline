import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

# import libraries
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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
    # take away the stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    
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
    Y_pred = model.predict(X_test)
    # display results
    print(classification_report(np.hstack(np.array(Y_test)),np.hstack(Y_pred)))
    
    

def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
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