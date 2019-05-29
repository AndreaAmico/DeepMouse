class Pspace(object):
    '''Scale points for bayes_opt library.
    '''
    def __init__(self, bounds={}):
        ''' Set the hyper-rectangle boundary for Bayes optimization.
        
        Set up the boundaries of the parameter space, defining both the min-max limits and the 
        scale: linear, log or descrete.
        
        The linear scale maps the min-max interval to [0, 1] and vice versa.
        The descrete scale maps the min-max integer interval to float [0, 1]. The opposite
        conversion will map the float [0, 1] to the integer [min, max] interval.
        The log scale maps the [min, max] interval to [0, 1]. The opposite conversion maps
        the [0, 1] interval to [10^min, 10^max] interval.
        
        Example:
            ps = Parameter_space({'a': (-2, 2, 'lin'),
                                  'b': (0, 20, 'descrete'),
                                  'c': (-8, 0, 'log')})
            
        
        Args:
            bounds (dict): boundaries dictionary: keys are the name of the parameters, values tuples
                containing (min parameter value, max parameter value, sampling scale). The sampling scale can be: 'linear',
                'logarithmic' or 'descrete'. For the logarithmic scale, the  min and max parameter values must be
                provided as base 10 exponents: (e.g. 4 means 1e4)
        
        '''
        self.bounds = bounds
        for k, v in self.bounds.items():
            assert (v[2] in ('lin', 'linear', 'int', 'integer', 'descrete', 'log', 'logarithmic')),\
            "Set the scale using 'lin', 'descrete', or 'log'"
        
    def _remap(self, x, in_min, in_max, out_min, out_max):
        ''' Linearly maps a float interval from [in_min, in_max] to [out_min, out_max].
        '''
        return (x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min
    
    def get_01(self):
        ''' Helper function which provides the pbounds parameter required by the BayesianOptimization class.
        Example:
            ps = Parameter_space({'a': (-2, 2, 'lin'),
                      'b': (0, 20, 'descrete'),
                      'c': (-8, 0, 'log')})
            ps.get_01() returns {'a': (0, 1), 'b': (0, 1), 'c': (0, 1)}
            
            A use case withing the bayes_opt library might be:
            optimizer = BayesianOptimization(
                f=test_function,
                pbounds=ps.get_01(),
                verbose=2,
                random_state=30)
        '''
        return {k:(0, 1) for k in self.bounds.keys()}
    
    def scaled(self, point):
        ''' Scale points from [min, max] interval to [0, 1].
        Args:
            point(dict): Dictionary describing a point in the parameter space [min,max]. Keys are the parameters names 
                and values correspond to the parameter value.
            
        Returns:
            new_point(dict): Input parameter space point mapped to [0, 1] range.
        '''
        new_point = {}
        for k, v in self.bounds.items():
            if v[2] in ('lin', 'linear'):
                new_point[k] = self._remap(point[k], v[0], v[1], 0, 1)
            elif v[2] in ('int', 'integer', 'descrete'):
                new_point[k] = self._remap(point[k], v[0], float(v[1]), 0, 1)
            elif v[2] in ('log', 'logarithmic'):
                new_point[k] = self._remap(np.log10(point[k]), v[0], v[1], 0, 1)
            else:
                raise ValueError("Set the scale using 'lin', 'descrete', or 'log'")
        return new_point
    
    def original(self, point):
        ''' Scale points from [0, 1] interval to [min, max].
        Args:
            point(dict): Dictionary describing a point in the parameter space [0,1]. Keys are the parameters names 
                and values correspond to the parameter value.
            
        Returns:
            new_point(dict): Input parameter space point mapped to [min, max] range.
        '''
        new_point = {}
        for k, v in self.bounds.items():
            if v[2] in ('lin', 'linear'):
                new_point[k] = self._remap(point[k], 0, 1, v[0], v[1])
            elif v[2] in ('int', 'integer', 'descrete'):
                new_point[k] = round(self._remap(point[k], 0, 1, v[0], v[1]))
            elif v[2] in ('log', 'logarithmic'):
                
                new_point[k] = 10**self._remap(point[k], 0, 1, v[0], v[1])
            else:
                raise ValueError("Set the scale using 'lin', 'descrete', or 'log'")                  
        return new_point
    
    def get_plot_scale(self, key):
        '''Get scale of a given parameter.
        
        Args:
            key(string): name of the parameter.
        
        Returns:
            scale(string): 'log' or 'linear' depending on the parameter scale .
        '''
        scale = self.bounds[key][2]
        return 'log' if scale in ('log', 'logarithmic') else 'linear'
    
    def array_to_original(self, arr, key_order):
        '''Maps a full array from [0, 1] to the [min, max].
        
        Args:
            arr(np.array): numpy array containing the parameters space coordinates as rows
            key_order(list): Parameter labels corresponding to the columns of the numpy array.
        
        Returns:
            new_arr(numpyy_array): numpy array with scaled coordinates to [min, max] interavl
                for each column og the input array.        
        '''
        new_arr = np.ones(arr.shape)
        for index, point in enumerate(arr):
            scaled_point = {k: point[i] for i, k in enumerate(key_order)}
            new_point = self.original(scaled_point)
            new_arr[index, :] = np.array([new_point[k] for k in key_order])
            
        return new_arr
    
    def optimizer_to_lists(self, optimizer, key_order):
        new_arr = np.ones([len(optimizer.res), len(key_order)+1])       
        for index, point in enumerate(optimizer.res):
            original_point = self.original(point['params'])            
            for param_index, param_name in enumerate(key_order):
                new_arr[index, param_index] = original_point[param_name]
            new_arr[index, -1] = point['target']
        return [new_arr[:,i] for i in range(new_arr.shape[1])]
    
    def optimizer_to_grid(self, optimizer, key_order):
        _ = optimizer.suggest(utility)
        
        x_span = np.linspace(0,1,50)
        y_span = np.linspace(0,1,50)
        
        xy_meshgrid = np.meshgrid(x_span, y_span, indexing='xy')
        xy_shape = xy_meshgrid[0].shape
        xy = np.column_stack([np.ravel(x) for x in xy_meshgrid])
        mean, sigma = optimizer._gp.predict(xy, return_std=True)
        new_key_order = [optimizer.space.keys.index(key) for key in key_order]
        xy_orig = ps.array_to_original(xy, optimizer.space.keys)
        x_grid = xy_orig[:, new_key_order[0]].reshape(xy_shape)
        y_grid = xy_orig[:, new_key_order[1]].reshape(xy_shape)
        mean = mean.reshape(xy_shape)
        sigma = sigma.reshape(xy_shape)
        
        return x_grid, y_grid, mean, sigma
    
    def plot_optimizer(self, optimizer, key_order, fig_axs=None):
        
        fig, axs = fig_ax if fig_axs else plt.subplots(1, 2, figsize=(10,5))
        
        X_PARAM = key_order[0]
        Y_PARAM = key_order[1]

        x_grid, y_grid, mean, sigma = ps.optimizer_to_grid(optimizer, key_order=[X_PARAM, Y_PARAM])
        X_points, Y_points, targets = ps.optimizer_to_lists(optimizer, key_order=[X_PARAM, Y_PARAM])

        axs[0].contourf(x_grid, y_grid, mean, 50, zorder=1)
        axs[0].set_title('Mean')
        axs[1].contourf(x_grid, y_grid, sigma, 50, zorder=1)
        axs[1].set_title('Sigma')

        x_max, y_max =  optimizer.max['params'][X_PARAM], optimizer.max['params'][Y_PARAM]
        
        for ax in axs:
            ax.scatter(X_points, Y_points, c=targets, cmap='magma', zorder=4, marker='d')
            ax.scatter(x_max, y_max, color='red', zorder=1, marker='o', alpha=0.5, s=150)

            ax.set_xlim(min(X_points), max(X_points))
            ax.set_ylim(min(Y_points), max(Y_points))
                       
            ax.set_xscale(ps.get_plot_scale(X_PARAM))
            ax.set_yscale(ps.get_plot_scale(Y_PARAM))

            ax.set_xlabel(f'{X_PARAM} (max: {x_max:.2e})')
            ax.set_ylabel(f'{Y_PARAM} (max: {y_max:.2e})')


# ps = Parameter_space({'a': (-2, 2, 'lin'),
#                       'b': (0, 20, 'descrete'),
#                       'c': (-8, 0, 'log')})


# point = {'a':15, 'b':1, 'c':1e-4}
# print(point)
# new_point = ps.scaled(point)
# print(new_point)
# new_point = ps.original(new_point)
# print(new_point)

# x_span = np.linspace(0,1,20)
# y_span = np.linspace(0,1,20)
# z_span = np.linspace(0,1,20)

# xx, yy, zz = np.meshgrid(x_span, y_span, z_span)
# xyz = np.concatenate(np.meshgrid(x_span, y_span, z_span)).reshape(-1, 3, order='F')

# aaa = ps.array_to_original(xyz, ['a', 'b', 'c'])