from config import test_all, over, mlai, ext, focus, deep, ext2, focus2

config_set = {
    'test_all': test_all.config,
    'over': over.config,
    'mlai': mlai.config,
    'ext': ext.config,
    'focus': focus.config,
    'deep': deep.config,
    'ext2': ext2.config,
    'focus2': focus2.config,
}

option_models = {
    'wrapper': ['tma', 'dta', 'trans'],
    'tune': [],
    'niv': [],
}


class ConfigSet:
    def __init__(self, config_set_name):
        self.config_set = config_set[config_set_name]
        self.dataset = self.config_set['dataset']

    def get_dataset_names(self):
        return self.dataset.keys()

    def get_model_names(self, dataset_name):
        return self.dataset[dataset_name].keys()

    def get_model(self, dataset_name, model_name):
        model = self.dataset[dataset_name][model_name].get('model')
        if model is not None:
            return model
        else:
            for check_name in test_all.config['dataset'][dataset_name]:
                if model_name.startswith(check_name):
                    return test_all.config['dataset'][dataset_name][check_name]['model']
            return None

    def get_search_space(self, dataset_name, model_name):
        search_space = self.dataset[dataset_name][model_name].get('space')
        if search_space is not None:
            return search_space
        else:
            check_model_name = model_name.replace('dt_', 'urf_').replace('mlai', 'glai')
            for check_name in test_all.config['dataset'][dataset_name]:
                if check_model_name.startswith(check_name):
                    return test_all.config['dataset'][dataset_name][check_name]['space']
            return None

    def get_default_params(self, dataset_name, model_name):
        params = self.dataset[dataset_name][model_name].get('params')
        if params is not None:
            return params
        else:
            check_model_name = model_name.replace('dt_', 'urf_').replace('mlai', 'glai')
            for check_name in test_all.config['dataset'][dataset_name]:
                if check_model_name.startswith(check_name):
                    return test_all.config['dataset'][dataset_name][check_name]['params']
            return None

    def is_enable(self, option, dataset_name=None, model_name=None):
        option_enable = self.config_set.get(option)
        if option_enable is not None:
            return option_enable
        elif dataset_name and model_name:
            option_enable = self.dataset[dataset_name][model_name].get(option)
            if option_enable is not None:
                return option_enable
            else:
                if model_name in option_models[option]:
                    return True
                else:
                    return False
        else:
            return False

    def get_option(self, option, dataset_name=None, model_name=None):
        option_method = self.config_set.get(option)
        if option_method is not None:
            return option_method
        elif dataset_name and model_name:
            return self.dataset[dataset_name][model_name].get(option)
        else:
            return None
