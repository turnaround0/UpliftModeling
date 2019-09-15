from models import model_dta_deep

config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'dta_deep': {'model': model_dta_deep},
        },
    },
    'wrapper': False,
    'niv': False,
    'tune': False,
}
