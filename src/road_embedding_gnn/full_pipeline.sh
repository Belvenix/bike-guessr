python3 download_graphs.py -t 
python3 download_graphs.py -v 
python3 transform_graphs.py -a -p ./data/data_train -o /app/data/data_transformed/train.bin
python3 transform_graphs.py -a -p ./data/data_val -o /app/data/data_transformed/validation.bin
python3 main.py 