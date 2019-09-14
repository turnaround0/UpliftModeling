from models import model_dt


def fit(x, y, t, **kwargs):
    kwargs.update({'method': 'ed'})
    return model_dt.fit(x, y, t, **kwargs)


def predict(obj, newdata, **kwargs):
    kwargs.update({'method': 'ed'})
    return model_dt.predict(obj, newdata, **kwargs)
