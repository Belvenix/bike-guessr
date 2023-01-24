import os
import time

def runStruc2vec():
    raise NotImplementedError
    # # categories = ['computer', 'humanreal', 'infrastructure', 'interaction', 'metabolic', 'coauthorshiip', 'humanonline']
    # # categories = ['humanreal', 'infrastructure', 'interaction', 'metabolic']
    # categories = ['test']
    # # numV = readExcel()
    # for category in categories:
    #     for root, dirs, files in os.walk('./data/' + category):
    #         for file in files:
    #             dataname = os.path.splitext(file)[0]
    #             print (dataname)
    #             os.system('/home/wyz/anaconda2/bin/python2.7 struc2vec/src/main.py --input '
    #                       'dividedata/' + category+'/'+dataname
    #                       + '.txt --output Struc2Vecemb/' + dataname
    #                       + '.emb --OPT3 true')