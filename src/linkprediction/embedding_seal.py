# https://github.com/muhanzhang/SEAL
import os
import time
 
def runSEAL():
    categories = ['coauthorship', 'computer', 'humanonline']
	#categories = ['metabolic']
    for category in categories:
        for root, dirs, files in os.walk('./data/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                start = time.time()
                for i in range(5, 31, 5):
                    os.system('/home/wyz/anaconda2/bin/python2.7 ./SEAL-master/Python/Main.py --data-name ' + dataname
                            + " --max-nodes-per-hop 100 --max-train-num 10000 --test-ratio "+str(float(i)/100))
                print ((time.time() - start) / 1000)




