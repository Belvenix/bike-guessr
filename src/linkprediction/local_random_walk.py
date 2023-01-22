def LRW(train, test, steps, lambda_):
    # Calculate LRW metrics and return AUC values
    import numpy as np
    deg = np.repeat(np.sum(train, axis=1), train.shape[1]).reshape(train.shape)
    train = np.nan_to_num(np.maximum(train / deg, 0))

    # find the transfer matrix
    I = np.eye(train.shape[0])
    
    # Generate the unit matrix
    sim = I
    for stepi in range(steps):
        # Random wandering iterations
        sim = (1-lambda_) * I + lambda_ * train.T @ sim
    sim = sim + sim.T
    
    # The similarity matrix is calculated
    train = np.array(train > 0, dtype=int)
    from sklearn.metrics import roc_auc_score
    
    # evaluate, calculate the AUC corresponding to this metric
    labels = test[np.triu_indices(test.shape[0], k=1)]
    scores = sim[np.triu_indices(sim.shape[0], k=1)]
    auc = roc_auc_score(labels, scores)
    return auc
