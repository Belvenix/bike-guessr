import networkx as nx
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def link_prediction_NB(train, test):
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    nb = BernoulliNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return auc


# Create a graph object
G = nx.Graph()

# Add edges to the graph
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

# Extract features for each node in the graph
X = []
for node in G.nodes():
    # Example feature: degree of the node
    degree = G.degree[node]
    X.append([degree])

# Generate labels for missing links
y = []
for node1 in G.nodes():
    for node2 in G.nodes():
        if not G.has_edge(node1, node2):
            y.append(1) # label 1 for missing links
        else:
            y.append(0) # label 0 for existing links

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the classifier object
clf = BernoulliNB()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier using ROC AUC
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC: ", roc_auc)