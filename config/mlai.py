from models import model_mlai

config = {
    'dataset': {
        'hillstrom': {
            'glai': {},
            'mlai_ed': {'model': model_mlai, 'class_weight': 'calculate'},
            'mlai_balanced': {'model': model_mlai,  'class_weight': 'balanced'},
        },
        'criteo': {
            'glai': {},
            'mlai_ed': {'model': model_mlai, 'class_weight': 'calculate'},
            'mlai_balanced': {'model': model_mlai,  'class_weight': 'balanced'},
        }
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
