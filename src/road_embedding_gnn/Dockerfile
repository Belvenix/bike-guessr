FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["bash", "full_pipeline.sh"]

# docker build -t road_embedding_gnn .
# docker run -it --name road_embedding_gnn road_embedding_gnn