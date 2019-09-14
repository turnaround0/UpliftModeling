from config import test_all, test

config_set = {
    'test_all': test_all.config,
    'test': test.config,
}

urf_methods = {
    'urf_ed': 'ed',
    'urf_kl': 'kl',
    'urf_chisq': 'chisq',
    'urf_int': 'int'
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
        return self.dataset[dataset_name][model_name]['model']

    def get_search_space(self, dataset_name, model_name):
        return self.dataset[dataset_name][model_name]['space']

    def get_default_params(self, dataset_name, model_name):
        params = self.dataset[dataset_name][model_name].get('params')
        if params is not None:
            return params
        else:
            return test_all.config['dataset'][dataset_name][model_name]['params']

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
