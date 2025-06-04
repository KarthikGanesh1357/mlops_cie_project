import pandas as pd
from nltk.corpus import stopwords

def clean_text(df, feature: str, custom_stop: list):
    '''
    Description:
        Clean text applying lower case, strip white spaces, removing
        unwanted characters, accents and stopwords
    Parameters:
        df: dataframe
            The dataset with the text feature.
        fearure: string
            The Text feature/column to clean.
    Return:
        Cleaned text.
    '''
    stop = stopwords.words('portuguese')
    stop.extend(custom_stop)
    # Clean text
    df[feature] = df[feature].str.lower()
    df[feature] = df[feature].str.strip()
    df[feature] = df[feature].str.replace(r'\d+', '')
    df[feature] = df[feature].str.replace(r'[^\w\s]', '')
    df[feature] = df[feature].str.findall(r'\w{3,}').str.join(' ')
    df[feature] = df[feature].str.normalize('NFKD').str.encode(
        'ascii', errors='ignore').str.decode('utf-8')
    df[feature] = df[feature].\
        apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))
    return df