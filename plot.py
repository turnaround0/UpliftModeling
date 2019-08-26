import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_fig5(var_sel_dict):
    plt.title('Variable selection')
    plt.xlabel('Amount of Variables')
    plt.ylabel('Qini Value')

    for model_name in var_sel_dict:
        if not var_sel_dict[model_name]:
            continue

        s_avg_qini = pd.DataFrame(var_sel_dict[model_name]).mean()[::-1]
        x_axis = range(len(s_avg_qini))

        plt.plot(x_axis, s_avg_qini, label=model_name)

    plt.legend()
    plt.show()


def plot_table6(qini_dict):
    titles = ['Qini', 'Qini top 30%', 'Qini Top 10%']
    index = qini_dict.keys()

    data = list()
    for model_name in qini_dict:
        qini_list = qini_dict[model_name]
        qini_all_list = [q['qini'] for q in qini_list]
        qini_30p_list = [q['qini_30p'] for q in qini_list]
        qini_10p_list = [q['qini_10p'] for q in qini_list]

        row = ['{:.5f} ({:.5f})'.format(np.mean(qini_all_list), np.std(qini_all_list)),
               '{:.5f} ({:.5f})'.format(np.mean(qini_30p_list), np.std(qini_30p_list)),
               '{:.5f} ({:.5f})'.format(np.mean(qini_10p_list), np.std(qini_10p_list))]
        data.append(row)

    df = pd.DataFrame(data=data, columns=titles, index=index)
    print(df)


def plot_fig7(qini_dict):
    plt.title('Qini Curve')
    plt.xlabel('Percentage')
    plt.ylabel('Qini Value')
    x_axis = range(0, 110, 10)

    is_draw_random = False
    for model_name in qini_dict:
        qini_list = qini_dict[model_name]

        if not is_draw_random:
            is_draw_random = True
            s_random_inc_gains = pd.DataFrame(data=[q['random_inc_gains'] for q in qini_list]).mean()
            plt.plot(x_axis, s_random_inc_gains, label='Random')

        s_inc_gains = pd.DataFrame(data=[q['inc_gains'] for q in qini_list]).mean()
        plt.plot(x_axis, s_inc_gains, label=model_name)

    plt.legend()
    plt.show()


def plot_fig8(qini_dict):
    best_model = 'tma'
    worst_model = 'dta'

    best_qini_list = qini_dict[best_model]
    worst_qini_list = qini_dict[worst_model]

    # Draw best curve
    # Draw worst curve


def plot_fig9(qini_dict):
    for n_fold in [0, 2]:
        plt.title('Information - Fold {}'.format(n_fold + 1))
        plt.xlabel('Proportion of population targeted (%)')
        plt.ylabel('Cumulative incremental gains (uplift %)')
        x_axis = range(0, 110, 10)

        is_draw_random = False
        for model_name in qini_dict:
            qini_data = qini_dict[model_name][n_fold]

            if not is_draw_random:
                is_draw_random = True
                plt.plot(x_axis, qini_data['random_inc_gains'], label='Random')

            plt.plot(x_axis, qini_data['inc_gains'], label=model_name)

        plt.legend()
        plt.show()
