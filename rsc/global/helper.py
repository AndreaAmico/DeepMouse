import numpy as np
import matplotlib.pyplot as plt


def get_color(index):
    color_list = ['#5374a2', '#a0a0a0']
    return color_list[index % len(color_list)]

def plot_scan_variable(df, scan_variable, ax=None, start=150, stop=190):

    data = []
    coord = []
    for index, row in df.iterrows():
        if not scan_variable in row.params.keys(): continue
        data.append(get_acc_between(row, start=150, stop=190))
        coord.append(row[scan_variable])
    data = np.array(data)

    fig, ax = plt.subplots(1, figsize=(5, 3)) if not ax else (ax.figure, ax)
    ax.errorbar(x=coord, y=data[:, 0], yerr=data[:, 1],
                fmt='h', color=get_color(1), label='training_accuracy')
    ax.errorbar(x=coord, y=data[:, 2], yerr=data[:, 3],
                fmt='o', color=get_color(2), label='test accuracy')
    ax.legend()
    ax.set(**{'xlabel': scan_variable, 'ylabel': 'accuracy'})
    return fig, ax


def get_history_data(fit_outs):
    accuracy_list = [x.history['acc'] for x in fit_outs]
    validation_accuracy_list = [x.history['val_acc'] for x in fit_outs]
    
    acc = np.mean(accuracy_list, 0)
    acc_err = np.std(accuracy_list, 0)
    
    val_acc = np.mean(validation_accuracy_list, 0)
    val_acc_err = np.std(validation_accuracy_list, 0)

    return acc, acc_err, val_acc, val_acc_err

def get_history_data_loss(fit_outs):
    accuracy_list = [x.history['loss'] for x in fit_outs]
    validation_accuracy_list = [x.history['val_loss'] for x in fit_outs]
    
    acc = np.mean(accuracy_list, 0)
    acc_err = np.std(accuracy_list, 0)
    
    val_acc = np.mean(validation_accuracy_list, 0)
    val_acc_err = np.std(validation_accuracy_list, 0)

    return acc, acc_err, val_acc, val_acc_err

def get_acc_between(df_row, start=150, stop=180):
    acc_array, acc_err_array, val_acc_array, val_acc_err_array = get_history_data(df_row.fit_outs)
    
    acc = np.mean(acc_array[start:stop])
    acc_err = np.mean(acc_err_array[start:stop])
    
    val_acc = np.mean(val_acc_array[start:stop])
    val_acc_err = np.mean(val_acc_err_array[start:stop])
    
    return acc, acc_err, val_acc, val_acc_err
    

def plot_mean_std(df_row, loss=False):
    fit_outs = df_row.fit_outs
    if loss:
        acc, acc_err, val_acc, val_acc_err = get_history_data_loss(fit_outs)
    else:
        acc, acc_err, val_acc, val_acc_err = get_history_data(fit_outs)

    line, = plt.plot(acc)
    plt.fill_between(range(acc.shape[0]), acc-acc_err, acc+acc_err, color=line.get_color(), alpha=0.3)
    
    line, = plt.plot(val_acc)
    plt.fill_between(range(val_acc.shape[0]), val_acc-val_acc_err, val_acc+val_acc_err, color=line.get_color(), alpha=0.3)
    
    plt.title(str(df_row.params))

