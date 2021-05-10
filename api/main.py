from fastapi import FastAPI

from api.predictor import Predictor

app = FastAPI()
predictor = Predictor()


@app.get('/')
async def root():
    return {'Hello': 'World'}


@app.post('/analyze/one/')
async def analyze_one(text: str):
    rank, rating = predictor.predict_one(text)
    return {'rank': rank}


@app.post('/analyze/one/deep/')
async def analyze_one_deep(text: str):
    rank, rating = predictor.predict_one(text)
    return {'rank': rank, 'rating': rating}


@app.post('/analyze/many/')
async def analyze_many(texts: list[str]):
    ranks, ratings = predictor.predict_many(texts)
    return {'ranks': ranks}


@app.post('/analyze/many/deep/')
async def analyze_many_deep(texts: list[str]):
    ranks, ratings = predictor.predict_many(texts)
    return {'ranks': ranks, 'ratings': ratings}


@app.post('/analyze/')
async def analyze(mode: str, deep: bool, data):
    function_name = f'predict_{mode}'
    function = getattr(predictor, function_name)
    short, long = function(data)
    return [short, long] if deep else [short]
