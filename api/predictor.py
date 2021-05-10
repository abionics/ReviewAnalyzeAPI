import numpy as np
import torch
from torch.utils.data import DataLoader

from ai.model import CustomLSTM
from ai.preprocessor import Preprocessor
from api.dataset import ReviewsDataset


class Predictor:

    def __init__(self):
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__preprocessor = Preprocessor()
        self.__model = CustomLSTM.load_from_file(self.__device, self.__preprocessor.capacity)
        self.__model.eval()

    def predict_one(self, text: str) -> (int, dict):
        encoded, length = self.__preprocessor.encode_sentence(text)
        tensor2d = torch.Tensor(encoded[np.newaxis])
        length1d = torch.tensor([length])

        tensor2d = tensor2d.long().to(self.__device)
        rating = self.__model(tensor2d, length1d)

        rank = torch.max(rating, 1)[1].item() + 1
        rating = rating.tolist()[0]
        rating = self.__rating_to_dict(rating)
        return rank, rating

    def predict_many(self, texts: list) -> (list, list):
        encoded = list(self.__preprocessor.encode_sentence(text) for text in texts)
        batch_size = min(len(encoded), 2000)
        dataset = ReviewsDataset(encoded)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        ranks = list()
        ratings = list()
        for tensor2d, length1d in dataloader:
            rating = self.__model(tensor2d, length1d)
            rank = torch.max(rating, 1)[1]
            ranks.extend(rank.tolist())
            ratings.extend(rating.tolist())
        ranks = list(rank + 1 for rank in ranks)
        ratings = list(self.__rating_to_dict(rating) for rating in ratings)
        return ranks, ratings

    @staticmethod
    def __rating_to_dict(rating: list) -> dict:
        return {
            i: round(r, 3)
            for i, r in enumerate(rating, start=1)
        }
