def RA(train, test, nodedegree):
    import numpy as np

    test = np.triu(test)
    test_num = np.count_nonzero(test)
    non_num = test_num

    test_data = np.zeros(test_num)
    i, j = np.where(test)
    for k in range(len(i)):
        cn = train[i[k], :] * train[j[k], :]
        if len(cn.shape) == 1:
            cn = np.reshape(cn, (-1, cn.shape[0]))
        x, y = np.where(cn)
        for l in range(len(y)):
            test_data[k] += 1 / nodedegree[y[l]]

    non_data = np.zeros(non_num)
    limiti = np.random.permutation(train.shape[0])[:int(2*np.ceil(np.sqrt(non_num)))]
    limitj = np.random.permutation(train.shape[1])[:int(2*np.ceil(np.sqrt(non_num)))]
    k = 0
    for i in limiti:
        if k >= non_num:
            break
        for j in limitj:
            if k >= non_num:
                break
            if (not train[i, j]) and (not test[i, j]) and (i != j):
                cn = train[i, :] * train[j, :]
                if len(cn.shape) == 1:
                    cn = np.reshape(cn, (-1, cn.shape[0]))
                x, y = np.where(cn)
                for l in range(len(y)):
                    non_data[k] += 1 / nodedegree[y[l]]
                k += 1

    from sklearn.metrics import roc_auc_score
    labels = np.concatenate((np.ones(test_num), np.zeros(test_num)))
    scores = np.concatenate((test_data, non_data))
    auc = roc_auc_score(labels, scores)
    return auc
