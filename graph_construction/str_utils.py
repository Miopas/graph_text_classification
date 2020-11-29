import re
import string
import pandas as pd

def read_csv(infile):
    try:
        df = pd.read_csv(infile)
    except:
        df = pd.read_csv(infile, lineterminator='\n')

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df
    #new_df = {'text':[], 'label':[]}
    #for (i, (text, label)) in enumerate(zip(df['text'], df['label'])):
    #    if pd.isna(label) or pd.isna(text): # skip NaN
    #        continue
    #    if int(label) not in [0, 1]:
    #        continue

    #    new_df['text'].append(text)
    #    new_df['label'].append(label)
    #return new_df

def preprocess(text):
    text = text.lower()
    text = re.sub(r'(\&amp\;)|(\&lt\;)', '', text)

    # Remove the users at the beginning and end of tweets
    text = re.sub(r'(^@\w+)|(@\w+$)', '', text)

    # Normalize users
    text = re.sub(r'@\w+', 'mary', text)

    # Remove urls
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", '', text)

    # Remove '#' in hashtags
    text = re.sub(r'#', '', text)

    # Remove the punctuations at the beginning and end of tweets.
    text = re.sub(r'(^[{}]+)|([{}]+$)'.format(string.punctuation+' ', string.punctuation+' '), '', text)

    text = text.strip()

    # Merge multiple blank space
    text = re.sub(r' +', ' ', text)

    return text

def process_digit(text):
    #if re.search(r'^(([0-9]+)|([0-9]+[\.\/][0-9]+))$', text) != None:
    #if text == '0.6':
    #    import pdb
    #    pdb.set_trace()
    if re.search('[0-9]+', text) != None:
        return '<NUM>'
    else:
        return text
