from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Main function for build model
def train_model (features, label, algo='mlp') :
    if features is None or label is None :
        return None
    elif algo is 'gb' :
        return build_gb_model(features, label)
    elif algo is 'mlp' :
        return build_mlp_model(features, label)

# Model builder functions goes here!
def build_mlp_model (features, label) :
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,))
    return clf.fit(features,label)

def build_gb_model (features, label, max_depth=7) :
    clf = GradientBoostingClassifier(max_depth=max_depth)
    return clf.fit(features, label)
