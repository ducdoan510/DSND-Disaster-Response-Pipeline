import sys
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    """

    :param database_filepath: filepath to sqlite db file
    :return: tuple (X, Y, category_names) with X: messages, Y: category value, category_names: names of categories
    """
    engine = create_engine("sqlite:///%s" % database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # drop id columns
    df = df.drop('id', axis=1)

    # get message column and assign to X
    X = df['message'].values
    
    # find category columns
    category_columns = df.select_dtypes(['int64'])
    category_names = category_columns.columns
    Y = category_columns.values

    return X, Y, category_names


def tokenize(text):
    """

    :param text: the string to tokenize
    :return: list of tokens from input text
    """
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def build_model():
    """

    :return: a pipeline model consisting of text preproessing and classifier
    """
    # for the purpose of this project, we only tune one parameter to save the training time
    parameters = {
        'clf__estimator__min_samples_split': [2, 5]
    }

    gridsearch_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    cv = GridSearchCV(gridsearch_pipeline, parameters, verbose=True)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model: a pipeline model classifier
    :param X_test: test independent variables
    :param Y_test: test dependent variables
    :param category_names: list of category names
    :return:
    """
    Y_cv_pred = model.predict(X_test)
    col_count = Y_cv_pred.shape[1]
    for col_idx in range(col_count):
        print('column:', category_names[col_idx])
        cat_pred = Y_cv_pred[:, col_idx]
        cat_test = Y_test[:, col_idx]
        print(classification_report(cat_test, cat_pred))


def save_model(model, model_filepath):
    """

    :param model: trained model to be saved
    :param model_filepath: path to pickle file to save the model
    :return:
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1], sys.argv[2]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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