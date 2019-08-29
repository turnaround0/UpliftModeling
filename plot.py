import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_fig5(dataset_name, var_sel_dict):
    fig, ax = plt.subplots()
    ax.set_title('Variable selection')
    ax.set_xlabel('Amount of Variables')
    ax.set_ylabel('Qini Value')

    for model_name in var_sel_dict:
        if not var_sel_dict[model_name]:
            continue

        # Input is reverse order, because it is about number of drop variables
        s_avg_qini = pd.DataFrame(var_sel_dict[model_name]).mean()[::-1]

        x_axis = np.arange(1, len(s_avg_qini) + 1)
        # x_ticks = np.arange(5, len(s_avg_qini) + 1, step=5)
        y_min = s_avg_qini.min() if s_avg_qini.min() < 0 else 0
        y_max = s_avg_qini.max() * 1.1

        # ax.set_xticks(x_ticks)
        ax.set_ylim([y_min, y_max])

        ax.plot(x_axis, s_avg_qini, label=model_name)

    ax.legend()
    plt.show()


def plot_table6(dataset_name, qini_dict):
    titles = ['Qini', 'Qini top 30%', 'Qini Top 10%']
    index = qini_dict.keys()

    if dataset_name == 'lalonde':
        factor = 1
    else:
        factor = 100

    data = list()
    for model_name in qini_dict:
        qini_list = qini_dict[model_name]
        qini_all_list = np.array([q['qini'] for q in qini_list]) * factor
        qini_30p_list = np.array([q['qini_30p'] for q in qini_list]) * factor
        qini_10p_list = np.array([q['qini_10p'] for q in qini_list]) * factor

        row = ['{:.5f} ({:.5f})'.format(np.mean(qini_all_list), np.std(qini_all_list)),
               '{:.5f} ({:.5f})'.format(np.mean(qini_30p_list), np.std(qini_30p_list)),
               '{:.5f} ({:.5f})'.format(np.mean(qini_10p_list), np.std(qini_10p_list))]
        data.append(row)

    df = pd.DataFrame(data=data, columns=titles, index=index)

    print('* Dataset:', dataset_name)
    print(df)


def plot_fig7(dataset_name, qini_dict):
    fig, ax = plt.subplots()
    ax.set_title('Qini Curve')
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Qini Value')

    x_axis = range(0, 110, 10)

    if dataset_name == 'lalonde':
        factor = 1
    else:
        factor = 100

    # Draw random line
    qini_list = qini_dict[next(iter(qini_dict))]
    s_random_inc_gains = pd.DataFrame(data=[q['random_inc_gains'] for q in qini_list]).mean() * factor
    ax.plot(x_axis, s_random_inc_gains, label='Random')

    # Draw line for each model
    for model_name in qini_dict:
        qini_list = qini_dict[model_name]
        s_inc_gains = pd.DataFrame(data=[q['inc_gains'] for q in qini_list]).mean() * factor
        ax.plot(x_axis, s_inc_gains, label=model_name)

    ax.legend()
    plt.show()


def plot_fig8(dataset_name, qini_dict):
    # Get average Qini value
    qini_values = []
    qini_models = list(qini_dict.keys())
    for model_name in qini_dict:
        qini_mean = np.mean([q['qini'] for q in qini_dict[model_name]])
        qini_values.append(qini_mean)

    # Find best and worst models
    best_model_idx = int(np.argmax(qini_values))
    worst_model_idx = int(np.argmin(qini_values))
    best_model = qini_models[best_model_idx]
    worst_model = qini_models[worst_model_idx]

    print('Best model:', best_model)
    print('Worst model:', worst_model)

    if dataset_name == 'lalonde':
        factor = 1
    else:
        factor = 100

    # Draw plot for each fold about best and worst models
    fig, axs = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    for plot_idx, model_name in enumerate([best_model, worst_model]):
        ax = axs[plot_idx]
        ax.set_title('Information - {}'.format(model_name))
        ax.set_xlabel('Proportion of population targeted (%)')
        if dataset_name == 'lalonde':
            ax.set_ylabel('Cumulative incremental gains (uplift)')
        else:
            ax.set_ylabel('Cumulative incremental gains (uplift %)')

        qini_list = qini_dict[model_name]
        x_axis = range(0, 110, 10)

        # Draw random uplift curve
        if dataset_name != 'lalonde':
            ax.plot(x_axis, np.array(qini_list[0]['random_inc_gains']) * factor, label='Random')
        else:
            # In case of lalonde dataset, each fold has different average.
            for idx in range(len(qini_list)):
                qini_data = qini_list[idx]
                ax.plot(x_axis, np.array(qini_data['random_inc_gains']) * factor, label='Random Fold {}'.format(idx + 1))

        # Draw model uplift curve
        for idx in range(len(qini_list)):
            qini_data = qini_list[idx]
            ax.plot(x_axis, np.array(qini_data['inc_gains']) * factor, label='Fold {}'.format(idx + 1))

        ax.legend()

    plt.show()


def plot_fig9(dataset_name, qini_dict):
    fig, axs = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    if dataset_name == 'lalonde':
        factor = 1
    else:
        factor = 100

    for idx, n_fold in enumerate([0, 2]):
        ax = axs[idx]
        ax.set_title('Information - Fold {}'.format(n_fold + 1))
        ax.set_xlabel('Proportion of population targeted (%)')
        if dataset_name == 'lalonde':
            ax.set_ylabel('Cumulative incremental gains (uplift)')
        else:
            ax.set_ylabel('Cumulative incremental gains (uplift %)')

        x_axis = range(0, 110, 10)
        qini_data = qini_dict[next(iter(qini_dict))][n_fold]
        ax.plot(x_axis, np.array(qini_data['random_inc_gains']) * factor, label='Random')

        for model_name in qini_dict:
            qini_data = qini_dict[model_name][n_fold]
            ax.plot(x_axis, np.array(qini_data['inc_gains']) * factor, label=model_name)

        ax.legend()

    plt.show()
