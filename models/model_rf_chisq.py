from models import model_rf


def fit(x, y, t, ntree=10, bagging_fraction=0.6, random_seed=1234, **kwargs):
    kwargs.update({'method': 'chisq'})
    return model_rf.fit(x, y, t, ntree, bagging_fraction, random_seed, **kwargs)


def predict(obj, newdata, **kwargs):
    kwargs.update({'method': 'chisq'})
    return model_rf.predict(obj, newdata, **kwargs)
