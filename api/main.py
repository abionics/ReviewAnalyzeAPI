from fastapi import FastAPI

from api.models import Text, Texts, Data
from api.predictor import Predictor

app = FastAPI()
predictor = Predictor()


@app.post('/analyze/one/')
async def analyze_one(text: Text):
    rank, rating = predictor.predict_one(text.text)
    return {'rank': rank}


@app.post('/analyze/one/deep/')
async def analyze_one_deep(text: Text):
    rank, rating = predictor.predict_one(text.text)
    return {'rank': rank, 'rating': rating}


@app.post('/analyze/many/')
async def analyze_many(texts: Texts):
    ranks, ratings = predictor.predict_many(texts.texts)
    return {'ranks': ranks}


@app.post('/analyze/many/deep/')
async def analyze_many_deep(texts: Texts):
    ranks, ratings = predictor.predict_many(texts.texts)
    return {'ranks': ranks, 'ratings': ratings}


@app.post('/analyze/')
async def analyze(data: Data):
    function_name = f'predict_{data.mode}'
    function = getattr(predictor, function_name)
    short, long = function(data.data)
    return [short, long] if data.deep else [short]
