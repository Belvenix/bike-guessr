python3 download_graphs.py -t 
python3 download_graphs.py -v 
python3 download_graphs.py -s
python3 transform_graphs.py --train -o train.bin
python3 transform_graphs.py --validation -o validation.bin
python3 transform_graphs.py --test -o test.bin
python3 train_classifiers.py
python3 visualize_predictions.py