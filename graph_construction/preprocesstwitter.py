"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import regex as re
#import re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


eyes = r"[8:=;]"
nose = r"['`\-]?"
re_patterns = {
'url': re.compile(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"),
'user': re.compile(r"@\w+"),
'smile' : re.compile(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes)),
'lolface' : re.compile(r"{}{}p+".format(eyes, nose)),
'sadface' : re.compile(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes)),
'neutralface' : re.compile(r"{}{}[\/|l*]".format(eyes, nose)),
'slash' : re.compile(r"/"),
'heart' : re.compile(r"<3"),
'number' : re.compile(r"[-+]?[.\d]*[\d]+[:,.\d]*"),
'hashtag' : re.compile(r"#\S+"),
'repeat' : re.compile(r"([!?.]){2,}"),
'elong' : re.compile(r"\b(\S*?)(.)\2{2,}\b"),
'allcaps' : re.compile(r"([A-Z]){2,}"),
}

def tokenize(text):
    # Different regex parts for smiley faces

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_patterns['url'].sub("<url>", text)
    text = re_patterns['user'].sub("<user>", text)
    text = re_patterns['smile'].sub("<smile>", text)
    text = re_patterns['lolface'].sub("<lolface>", text)
    text = re_patterns['sadface'].sub("<sadface>", text)
    text = re_patterns['neutralface'].sub("<neutralface>", text)
    text = re_patterns['slash'].sub(" / ", text)
    text = re_patterns['heart'].sub("<heart>", text)
    text = re_patterns['number'].sub("<number>", text)
    text = re_patterns['hashtag'].sub(hashtag, text)
    text = re_patterns['repeat'].sub(r"\1 <repeat>", text)
    text = re_patterns['elong'].sub(r"\1\2 <elong>", text)

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_patterns['allcaps'].sub(allcaps, text)

    return text.lower()


def process_tweets(text):
    return tokenize(text)

def process_reddit(text):
    text = tokenize(text)
    if '[removed]' in text or '[deleted]' in text:
        return ''
    return text

if __name__ == '__main__':
    _, text = sys.argv
    if text == "test":
        text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    tokens = tokenize(text)
    print(tokens)
