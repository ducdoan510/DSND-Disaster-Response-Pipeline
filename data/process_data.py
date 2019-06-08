import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """

    :param messages_filepath: path to csv file with message data
    :param categories_filepath: path to csv file with category data
    :return: a dataframe merging the two input datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """

    :param df: the dataframe with message and category info to be cleaned
    :return: cleaned dataframe
    """
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # extract a list of new column names for categories.
    category_colnames = row.map(lambda category: category[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).map(lambda cat: cat[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """

    :param df: the clean dataframe to be saved
    :param database_filename: path to sqlite db file
    :return:
    """
    engine = create_engine('sqlite:///%s' % database_filename)
    table_name = database_filename.replace('.db', '')
    df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=500)
    rows = engine.execute('select * from %s' % table_name).fetchall()
    print(pd.DataFrame(rows, columns=df.columns))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:4]

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