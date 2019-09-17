# UpliftModeling
## Environment
This program was tested on Anaconda3 / python 3.7 environment.<br>
Please install required python libraries.
``` console
$ pip install -r requirements.txt
```

## How to execute
### Example of test for the uplift article
At first, we tested several experiments on the uplift article.<br>
Test_all configuration was used for the experiment.<br>
You can execute it by below command.
``` console
$ python uplift.py -s test_all
```
After -s option, please put test configuration name.<br>
After this execution, you can see Qini table.<br>
The test result will be saved on /output/*.json files.

If you want to plot the results, please execute below command.
``` console
$ python uplift.py -s test_all -d
```
-d option means 'display the results'.<br>
The program will load /output/*.json files,<br>
and it will display some plots with the loaded data.

## Code structure
### 1. Configuration (/config)
/config/config.py file has definitions of configuration sets.
``` python
config_set = {
    'test_all': test_all.config,
    'test': test.config,
    'over': over.config,
    'mlai': mlai.config,
    'ext': ext.config,
    'focus': focus.config,
    'deep': deep.config,
    'ext2': ext2.config,
    'focus2': focus2.config,
}
```
Each configuration was defined on /config/\*.py files.<br>
If you skip to write detailed configuration on /config/\*.py files,<br>
its configuration values will come from default values.<br>
(Default values: default.py + test_all.py)
``` python
config = {
    'dataset': {
        'hillstrom': {
            'dta': {},
            'dta_deep': {'model': model_dta_deep, 'params': {'method': 'logistic'}},
        },
```

If you want to make one more configuration set,<br>
please create /config/{filename}.py file and,<br>
modify config_set on /config/config.py file.

### 2. Input data (/input)
Input csv files located on /input directory.<br>
They will be loaded by dataset.py file.

### 3. Models (/models)
Each model is defined by one model python file.<br>
So, if you want to create new model,<br>
please create /models/{filename}.py file and add the model on configuration file.
``` config
config = {
    'dataset': {
        'hillstrom': {
            'dt_ed': {'model': model_dt_ed, 'params': params_tree_hillstrom},
            'dt_ed_ext': {'model': model_dt_ed_ext, 'params': params_tree_hillstrom,
                          'max_round': 4, 'u_list': [2.0, 1.0, 0.5, -float('INF')]},
            'tma': {},
            'tma_ext': {'model': model_tma_ext, 'max_round': 3, 'u_list': [1.0, 0.7, -float('INF')]},
        },
```

Each model should provide fit and predict methods.<br>
They will be called on uplift.py file.

Some models require extra parameters.<br>
In that case, set_params should be implemented.

### 4. Output (/output)
Test result will be saved on output folder by json format.<br>
Whenever you want after test, you can see the test result by -d option.

### 5. Over sampling method (/over)
This folder provides over sampling methods.<br>
Each file provides only one over sampling method.<br>
If you want to create new over sampling method,<br>
please add /over/{filename}.py file and modify configuration file.
``` python
config = {
    'dataset': {
        'hillstrom': {
            'tma': {'params': params_logistic},
            # 'tma_simple': {'over_sampling': simple.over_sampling, 'params': params_logistic},
            'tma_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'tma_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic},
            'tma_gan2': {'over_sampling': gan.over_sampling2, 'params': params_logistic},
        },
        ...
```

### 6. Tree (/tree)
Tree algorithm located on this folder.<br>
In model files, they only use these /tree/*.py files to make tree and predict.<br>
If you see build_tree method of /tree/tree.py file,<br>
you can find tree extraction method codes.
 
### 7. Others (/)
uplift.py: main method<br>
dataset.py: load dataset and load/save output json files<br>
measure.py: measure performance and get Qini values<br>
plot.py: display test results by graphs and tables<br>
preprocessing.py: pre-process input data<br>
tune.py: codes related with parameter tuning<br>
uplift.py: return uplift value of given dataset
