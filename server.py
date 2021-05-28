import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel # , validator
from typing import Optional
from model import model_loader_bert, model_predictions_bert, model_loader_gpt, model_predictions_gpt
import torch

# import os
#
# print(os.getcwd())
# print(os.listdir(path="."))

# App creation and model loading
app = FastAPI()
model_gpt_title, tokenizer_gpt_title = model_loader_gpt(path='s_line_512_title', device=torch.device("cpu"), fp16=False)
model_gpt_descr, tokenizer_gpt_descr = model_loader_gpt(path='s_line_512_descr', device=torch.device("cpu"), fp16=False)

model_bert_title, tokenizer_bert_title = model_loader_bert(path='Bert_ru_512_title.bin', device=torch.device("cpu"),
                                                           fp16=False)
model_bert_descr, tokenizer_bert_descr = model_loader_bert(path='Bert_ru_512_descr.bin', device=torch.device("cpu"),
                                                           fp16=False)
print('Model loader')


class IrisSpecies(BaseModel):
    """
    Input features validation for the ML model
    """
    news: Optional[str] = ''
    max_len_title: Optional[int] = 32
    max_len_descr: Optional[int] = 128
    k: Optional[int] = 50
    p: Optional[float] = 0.95
    num_return_sequences: Optional[int] = 3
    threshold: Optional[float] = 0.0

    # class Config:
    #     validate_assignment = True
    #
    # @validator('max_len_title')
    # def set_max_len_title(cls, max_len_title):
    #     return max_len_title or 32
    #
    # @validator('max_len_descr')
    # def max_len_descr(cls, max_len_descr):
    #     return max_len_descr or 64


@app.post('/predict')
def predict(iris: IrisSpecies):
    """
    :param iris: input data from the post request
    :return: predicted iris type
    """
    generated_sequences_title = model_predictions_gpt(model_gpt_title, tokenizer_gpt_title, torch.device("cpu"),
                                                      iris.max_len_title, [iris.news + ' =>'], temperature=0.9, k=iris.k, p=iris.p,
                                                      repetition_penalty=1.0, num_return_sequences=iris.num_return_sequences)
    generated_sequences_descr = model_predictions_gpt(model_gpt_descr, tokenizer_gpt_descr, torch.device("cpu"),
                                                      iris.max_len_descr, [iris.news + ' =>'], temperature=0.9, k=iris.k, p=iris.p,
                                                      repetition_penalty=1.0, num_return_sequences=iris.num_return_sequences)

    quality_title = model_predictions_bert(model_bert_title, tokenizer_bert_title, device=torch.device("cpu"),
                                           news=iris.news, data=generated_sequences_title, threshold=iris.threshold)
    quality_descr = model_predictions_bert(model_bert_descr, tokenizer_bert_descr, device=torch.device("cpu"),
                                           news=iris.news, data=generated_sequences_title, threshold=iris.threshold)

    prediction_title = [f'{i}: {j}' for i, j in zip(quality_title, generated_sequences_title)]
    prediction_descr = [f'{i}: {j}' for i, j in zip(quality_descr, generated_sequences_descr)]

    return {
        "prediction_title": prediction_title,
        "prediction_descr": prediction_descr
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
