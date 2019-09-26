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
    'test_all': test_all.config,  # Experiment for the article
    'over': over.config,          # Over-sampling test
    'mlai': mlai.config,          # Modified GLAI test
    'ext': ext.config,            # Tree extraction method
    'focus': focus.config,        # Tree focus method
    'deep': deep.config,          # DNN test (with DTA or without DTA)
    'dis': dis.config,            # GAN Discriminator test
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
            'deep': {'model': model_dta_deep,
                     'params': {'method': 'logistic', 'lr': 1e-3, 'epochs': 100,
                                'batch_size': 256, 'decay': 1e-2}},
            'dta_deep': {'model': model_dta_deep,
                         'params': {'method': 'logistic', 'lr': 1e-3, 'epochs': 100,
                                    'batch_size': 256, 'decay': 1e-2}},
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
                          'max_round': 4, 'p_value': 0.2},
        },
        ...
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
            'dta': {'params': params_logistic},
            'dta_smote': {'over_sampling': smote.over_sampling, 'params': params_logistic},
            'dta_gan': {'over_sampling': gan.over_sampling, 'params': params_logistic,
                        'params_over': params_gan_hillstrom},
            'dta_tfgan': {'over_sampling': tfgan.over_sampling, 'params': params_logistic,
                          'params_over': params_tfgan_hillstrom},
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
/dataset: load dataset and preprocessing<br>
/deep: deep learning code (DNN, GAN, TFGAN)<br>
/experiment: measure performance, get Qini values and display test results<br>
/tune: tuning algorithm codes (NIV, general wrapper approach, parameter tuning)<br>
/utils: helper methods like loading or saving json<br>
