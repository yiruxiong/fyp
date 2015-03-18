def predict(self, features):
    '''Dot-product the features and current weights and return the best class.'''
    scores = defaultdict(float)
    for feat in features:
        if feat not in self.weights:
            continue
        weights = self.weights[feat]
        for clas, weight in weights.items():
            scores[clas] += weight
    # Do a secondary alphabetic sort, for stability
    return max(self.classes, key=lambda clas: (scores[clas], clas))