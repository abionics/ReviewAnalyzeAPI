import typing

from pydantic import BaseModel


class Text(BaseModel):
    text: str


class Texts(BaseModel):
    texts: list[str]


class Data(BaseModel):
    mode: str
    deep: bool
    data: typing.Union[str, list[str]]
