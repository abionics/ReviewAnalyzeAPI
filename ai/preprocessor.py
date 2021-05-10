import json
import re
import string

import en_core_web_sm
import numpy as np

from config import word2vec_FILENAME


class Preprocessor:

    def __init__(self):
        self.__english = en_core_web_sm.load()
        self.__word2vec = self.__load_word2vec()

    @staticmethod
    def __load_word2vec() -> dict:
        with open(word2vec_FILENAME, 'r') as file:
            return json.load(file)

    @property
    def capacity(self) -> int:
        return len(self.__word2vec)

    def encode_sentence(self, text: str, size: int = 150) -> (list, int):
        """
        Encode reviews to array of ints
        :param text: text to encode
        :param size: max encoded size
        :return: list of ints and encoded size
        """
        tokenized = self.__tokenize(text)
        encoded = np.zeros(size, dtype=int)
        unknown_symbol = self.__word2vec['UNK']
        enc1 = np.array([
            self.__word2vec.get(word, unknown_symbol)
            for word in tokenized
        ])
        length = min(size, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

    def __tokenize(self, text: str) -> list:
        """
        Tokenize in order to clean text
        :param text: text to tokenize
        :return: list of tokens
        """
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        words = regex.sub(' ', text.lower())
        words = re.sub(r'\s+', ' ', words.strip(), flags=re.UNICODE)
        return [token.text for token in self.__english.tokenizer(words)]
