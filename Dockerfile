FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /tech-practice

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/tech-practice"
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

RUN mkdir -p /models

RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained( \
    'openai/clip-vit-base-patch32' \
).save_pretrained('/models/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained( \
    'openai/clip-vit-base-patch32' \
).save_pretrained('/models/clip-vit-base-patch32')"

CMD ["python", "app/web.py"]