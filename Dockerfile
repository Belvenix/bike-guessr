FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY src .

RUN dos2unix full_pipeline.sh

CMD ["bash", "full_pipeline.sh"]

# docker build -t road_embedding_gnn .
# docker run -it --name road_embedding_gnn road_embedding_gnn