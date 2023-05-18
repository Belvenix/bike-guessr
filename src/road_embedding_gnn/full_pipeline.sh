python3 download_graphs.py --train
python3 download_graphs.py --validation
python3 transform_graphs.py -a -p ./data/data_train -o .data/data_transformed
pytohn3 transform_graphs.py -a -p ./data/data_val -o .data/data_transformed
python3 main.py