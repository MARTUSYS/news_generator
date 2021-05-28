import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import model_loader_bert, model_predictions_bert, model_loader_gpt, model_predictions_gpt
import torch

# import os

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


class IrisSpecies(BaseModel):
    """
    Input features validation for the ML model
    """
    news: str
    max_len_title: int
    max_len_descr: int


@app.post('/predict')
def predict(iris: IrisSpecies):
    """
    :param iris: input data from the post request
    :return: predicted iris type
    """
    generated_sequences_title = model_predictions_gpt(model_gpt_title, tokenizer_gpt_title, torch.device("cpu"),
                                                      iris.max_len_title, [iris.news], temperature=0.9, k=50, p=0.95,
                                                      repetition_penalty=1.0, num_return_sequences=3)
    generated_sequences_descr = model_predictions_gpt(model_gpt_descr, tokenizer_gpt_descr, torch.device("cpu"),
                                                      iris.max_len_descr, [iris.news], temperature=0.9, k=50, p=0.95,
                                                      repetition_penalty=1.0, num_return_sequences=3)

    quality_title = model_predictions_bert(model_bert_title, tokenizer_bert_title, device=torch.device("cpu"),
                                           news=iris.news, data=generated_sequences_title, threshold=0)
    quality_descr = model_predictions_bert(model_bert_descr, tokenizer_bert_descr, device=torch.device("cpu"),
                                           news=iris.news, data=generated_sequences_title, threshold=0)

    prediction_title = [f'{i}: {j}' for i, j in zip(quality_title, generated_sequences_title)]
    prediction_descr = [f'{i}: {j}' for i, j in zip(quality_descr, generated_sequences_descr)]

    return {
        "prediction_title": prediction_title,
        "prediction_descr": prediction_descr
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
