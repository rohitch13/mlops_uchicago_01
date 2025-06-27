from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import io
import requests

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def model_pipeline(text: str, image: Image.Image):
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <body>
            <h2>ViLT VQA API is Running</h2>
            <a href="/upload">Go to Upload Page</a>
        </body>
    </html>
    """


@app.get("/upload", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <body>
            <h2>ViLT VQA App</h2>
            <form action="/ask" enctype="multipart/form-data" method="post">
                <label>Question:</label><br>
                <input type="text" name="text" value="What are the colors of the cats?" /><br><br>
                <label>Upload Image:</label><br>
                <input type="file" name="image" accept="image/*" /><br><br>
                <input type="submit" value="Ask Question" />
            </form>
            <br>
            <h3>Or use demo image:</h3>
            <form action="/demo" method="get">
                <input type="submit" value="Run Demo" />
            </form>
        </body>
    </html>
    """

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(text: str = Form(...), image: UploadFile = Form(...)):
    content = await image.read()
    image_pil = Image.open(io.BytesIO(content)).convert("RGB")
    answer = model_pipeline(text, image_pil)
    return f"""
    <html>
        <body>
            <h2>Question: {text}</h2>
            <h3>Predicted Answer: {answer}</h3>
            <a href="/upload">Back</a>
        </body>
    </html>
    """

@app.get("/demo", response_class=HTMLResponse)
async def run_demo():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    text = "What are the colors of the cats?"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    answer = model_pipeline(text, image)
    return f"""
    <html>
        <body>
            <h2>Demo Question: {text}</h2>
            <img src="{url}" width="400"/><br><br>
            <h3>Predicted Answer: {answer}</h3>
            <a href="/upload">Back</a>
        </body>
    </html>
    """

