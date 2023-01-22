from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF

def MF(train, test, k=8, SGD=False):
    if SGD:
        # Use adaptive SGD with separate validation set
        from sklearn.model_selection import train_test_split
        train, validation = train_test_split(train, test_size=0.1)
    else:
        validation = None
    nmf = NMF(n_components=k)
    nmf.fit(train)
    pred = nmf.predict(test)
    if validation is not None:
        pred_validation = nmf.predict(validation)
    else:
        pred_validation = None
    thisauc = roc_auc_score(test, pred)
    return thisauc, pred, pred_validation