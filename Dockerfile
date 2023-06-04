FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN apt-get update

RUN apt-get install -y --no-install-recommends dos2unix

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY src .

RUN dos2unix full_pipeline.sh

CMD ["bash", "full_pipeline.sh"]

# docker build -t road_embedding_gnn .
# docker run -it --name road_embedding_gnn road_embedding_gnn