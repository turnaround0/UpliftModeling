from models import model_glai

config = {
    'dataset': {
        'hillstrom': {
            'glai': {},
            # 'mlai': {'model': model_mlai, 'mlai_values': [0, 0.3, 0.5, 0.7, 1], 'class_weight':'calculate'},
            'mlai_ed': {'model': model_glai, 'class_weight':'calculate'},
            # 'mlai_balanced': {'model': model_mlai, 'mlai_values': [0, 0.3, 0.5, 0.7, 1], 'class_weight':'balanced'},
            'mlai_balanced': {'model': model_glai,  'class_weight':'balanced'},
        },
        'criteo': {
            'glai': {},
             # 'mlai': {'model': model_mlai, 'mlai_values': [0, 0.3, 0.5, 0.7, 1], 'class_weight':'calculate'},
            'mlai_ed': {'model': model_glai, 'class_weight':'calculate'},
            # 'mlai_balanced': {'model': model_mlai, 'mlai_values': [0, 0.3, 0.5, 0.7, 1], 'class_weight':'balanced'},
            'mlai_balanced': {'model': model_glai,  'class_weight':'balanced'},
        }
    },
    'wrapper': False,
    'niv': True,
    'tune': False,
}
