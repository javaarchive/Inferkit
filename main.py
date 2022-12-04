import config

from typing import Union

from fastapi import FastAPI

from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

from transformers import pipeline

from pydantic import BaseModel

@app.get("/")
def read_root():
    return {"Hello": "World"}


if config.TEXT_ZEROSHOT_CLASSIFY:
    classify_model = pipeline(model=config.TEXT_ZEROSHOT_CLASSIFY)

    class TextClassifyTask(BaseModel):
        text: str
        labels: list[str] = config.DEFAULT_LABELS

    @app.post("/api/text_classify")
    def categorize(task: TextClassifyTask):
        return {"result": classify_model(task.text,candidate_labels=task.labels),"ok": True}

if config.ENABLE_LATEX_OCR:
    from pix2tex.cli import LatexOCR
    from PIL import Image
    from io import BytesIO
    model = LatexOCR()
    @app.post('/api/latex_ocr')
    async def predict_from_file(file: bytes = File(...), resize: bool = False) -> str:  # , size: str = Form(...)
        # taken from https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/api/app.py
        global model
        #size = tuple(int(a) for a in size.split(','))
        image = Image.open(BytesIO(file))
        return {"result":model(image, resize=resize),"ok": True}

if config.TEXT_SUMMARIZER:
    summarizer = pipeline("summarization", model = config.TEXT_SUMMARIZER)
    class SummarizeTask(BaseModel):
        text: str
        max_length: int = 100
        min_length: int = 30
    @app.post("/api/text_summarize")
    def categorize(task: SummarizeTask):
        return {"result": summarizer(task.text,min_length = task.min_length, max_length = task.max_length),"ok": True}

if config.TEXT_QUESTION_ASK:
    qa_model = pipeline(model = config.TEXT_QUESTION_ASK)
    class QuestionAnswerTask(BaseModel):
        question: str
        context: str
    @app.post("/api/text_question_answer")
    def ask(task: QuestionAnswerTask):
        return {"result": qa_model(question = task.question,context = task.context),"ok": True}

if config.TEXT_GENERATOR:
    generator = pipeline("text-generation",model = config.TEXT_GENERATOR)
    class GenerateTask(BaseModel):
        text: str
        full_text: bool = False
    @app.post("/api/text_generate")
    def generate(task: GenerateTask):
        return {"result": generator(task.text,return_full_text=task.full_text),"ok": True}