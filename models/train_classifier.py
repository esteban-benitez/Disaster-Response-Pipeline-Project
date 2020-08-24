# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from pprint import pprint
from time import time
import logging


def load_data(database_filepath):

    """load_data(database_filepath) returns X, Y, category_names after .db file is loaded to pandas DataFrame"""

    engine = create_engine('sqlite:///' + database_filepath, echo = False)

    df = pd.read_sql_table(database_filepath, con = engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):

    """tokenize(text) method utilized to split a given text into individual words"""

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():

    """creates a Pipeline() object containing CountVectorizer, TfidfTransformer(), and RandomForestClassifier(), returns the Pipeline() object"""

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    """evaluate_model(model, X_test, Y_test, category_names) and prints the classification_report"""

    Y_pred = model.predict(X_test)
    for col in range(36):
        print(Y_test.columns[col])
        print(classification_report(Y_test.iloc[:,col], Y_pred[:,col]))
        print('-----------------------------------------------------')


def save_model(model, model_filepath):

    """save_model(model, model_filepath) as a pickle file"""

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def gridSearch(pipeline, X_train, Y_train):
    parameters = {
        'vect__ngram_range': ((1, 2), (1, 3)),
        'clf__estimator__criterion': ['gini', 'entropy'],
        'tfidf__smooth_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'vect__max_features': [1, 5, 10],
        'clf__estimator__n_estimators': [1, 3, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    cv.fit(X_train, Y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % cv.best_score_)
    print("Best parameters set:")
    best_parameters = cv.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    


def main():
    
    """ static main() method utilized to initialize ML Pipeline """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('X: ', X)
        print('Y: ', Y)
        print("category_names: ", category_names)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Grid Searching hyperparameters...')
        gridSearch(build_model(), X_train, Y_train)

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