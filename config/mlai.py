from models import model_mlai

config = {
    'dataset': {
        'hillstrom': {
            'glai': {},
            'mlai': {'model': model_mlai, 'mlai_values': [0, 0.3, 0.5, 0.7, 1]},
        },
    },
    'wrapper': False,
    'niv': False,
    'tune': False,
}
