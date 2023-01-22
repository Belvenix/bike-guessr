import matlab.engine
import numpy as np

def SBM(train, test, k):

    eng = matlab.engine.start_matlab()

    # To run specific matlab script it should be on the path the script is run on
    # ie eng.<script-name> ie eng.myscript

    # Go to specific path so that it can run the matlab file
    eng.cd("src/software/WSBM_v1-2.2", nargout=0)
    #eng.SBM(np.matrix([[0, 1, 1, 0],[1, 0, 1, 0],[1, 1, 0, 1],[0, 0, 1, 0]]), np.matrix([[0, 1, 0, 0],[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0]]), 12)
    eng.SBM(train, test, k)