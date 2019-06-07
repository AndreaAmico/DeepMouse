import numpy as np

def pre_process_data(X, y, grid):
    # relative mouse movement
    X_diff = X[:,:-1,:].copy()
    X_diff[:,:,0] = np.diff(X[:,:,0])
    X_diff[:,:,1] = np.diff(X[:,:,1])
    
    # remove batches with no mouse activity
    sigma_x = np.std(X_diff[:,:,0], axis=1)
    sigma_y = np.std(X_diff[:,:,1], axis=1)
    mask = (sigma_x>grid['sigma_cut'][1])*(sigma_y>grid['sigma_cut'][1])

    X_filt = X_diff[mask]
    y_filt = y[mask]

    # cast to float
    X_filt = X_filt.astype('float')

    # normalize data
    x_std = grid['x_std'][1] # np.mean(np.std(X_filt[:,:,0], axis=1))
    y_std = grid['y_std'][1] # np.mean(np.std(X_filt[:,:,1], axis=1))

    X_filt[:,:,0] = X_filt[:,:,0] / x_std
    X_filt[:,:,1] = X_filt[:,:,1] / y_std
    
    return X_filt, y_filt
