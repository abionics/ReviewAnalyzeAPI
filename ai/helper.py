import pickle
import re
import string
import tarfile

import en_core_web_sm
import numpy as np
import pandas as pd
import requests

DATASET_URL = 'http://hidra.lbd.dcc.ufmg.br/datasets/yelp_2015/original/yelp_review_full_csv.tar.gz'
DATASET_ARCHIVE_NAME = 'dataset.tar.gz'
DATASET_FOLDER_NAME = 'dataset'

ENGLISH = en_core_web_sm.load()


# Save/load data from file to increase speed when rerun this notebook
def save(data, filename: str = 'data.pkl'):
    with open(f'pickles/{filename}', 'wb') as file:
        pickle.dump(data, file)


def load(filename: str = 'data.pkl'):
    with open(f'pickles/{filename}', 'rb') as file:
        return pickle.load(file)


# Download dataset
def download_file(url: str, filename: str, chunk_size: int = 2 ** 15):
    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        with open(filename, 'wb') as file:
            for chunk in request.iter_content(chunk_size=chunk_size):
                file.write(chunk)


def extract_archive(archive_name: str, folder_name: str):
    with tarfile.open(archive_name) as tar:
        tar.extractall(path=folder_name)


# download_file(DATASET_URL, DATASET_ARCHIVE_NAME)
# extract_archive(DATASET_ARCHIVE_NAME, DATASET_FOLDER_NAME)

# Read dataset
test = pd.read_csv(f'{DATASET_FOLDER_NAME}/yelp_review_full_csv/test.csv', header=None)
train = pd.read_csv(f'{DATASET_FOLDER_NAME}/yelp_review_full_csv/train.csv', header=None)

test = test.rename(columns={0: 'label', 1: 'review'})
train = train.rename(columns={0: 'label', 1: 'review'})

print(len(test), len(train))
train.head()


# Tokenize in order to clean text
def tokenize(text: str):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    words = regex.sub(' ', text.lower())
    words = re.sub(r'\s+', ' ', words.strip(), flags=re.UNICODE)
    return [token.text for token in ENGLISH.tokenizer(words)]


# counts = Counter()
# for index, row in train.iterrows():
#     if index % 100 == 0:
#         percent = 100 * index // len(train)
#         print(f'{percent}%')
#     counts.update(tokenize(row['description']))
# save(counts, 'counts.pkl')
counts = load('counts.pkl')

# Check words with spaces (must be empty)
for word in list(counts):
    if ' ' in word:
        print(word)
        print(counts[word])

# Create vocabulary
counts_most = counts.most_common(2000)
word2vec = {'': 0, 'UNK': 1}
words = ['', 'UNK']
for word, freq in counts_most:
    word2vec[word] = len(words)
    words.append(word)


# Encode reviews to array of int
def encode_sentence(text: str, word2vec: dict, size: int = 150):
    tokenized = tokenize(text)
    encoded = np.zeros(size, dtype=int)
    enc1 = np.array([word2vec.get(word, word2vec['UNK']) for word in tokenized])
    length = min(size, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


# train['encoded'] = train['review'].apply(lambda x: np.array(encode_sentence(x, word2vec), dtype=object))
# save(train, 'train.pkl')
train = load('train.pkl')

# Find and drop reviews with zero word length
indexes = []
for index, row in train.iterrows():
    if row['encoded'][0][0] == 0:
        indexes.append(index)
print(train.iloc[indexes].head())
# train.drop(train.index[indexes], inplace=True)

# test['encoded'] = test['review'].apply(lambda x: np.array(encode_sentence(x, word2vec), dtype=object))
# save(test, 'test.pkl')
test = load('test.pkl')
