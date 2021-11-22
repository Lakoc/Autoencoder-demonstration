import numpy as np


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def remove_values_from_list_to_float(the_list, val):
    return [float(value) for value in the_list if value != val]


def load_3d_arr_from_string(arr):
    arr = arr.replace('[', '').replace(']', '').split('\n')
    count = arr.count('') + 1

    arr = remove_values_from_list(arr, '')
    group_size = len(arr) // count
    groups = [remove_values_from_list_to_float(val.split(' '), '') for group in range(count) for val in
              arr[group * group_size: (group + 1) * group_size]]
    groups = [groups[group * group_size: (group + 1) * group_size] for group in range(count)]
    return np.array(groups)


def normalize_config(config):
    config['variances'] = load_3d_arr_from_string(config['variances'])
    config['means'] = load_3d_arr_from_string(config['means'])[0, :]
    config['counts'] = load_3d_arr_from_string(config['counts'])[0, 0, :]
    config['layers'] = load_3d_arr_from_string(config['layers'])[0, 0, :]
    config['layer'] = int(config['layer'])
    config['batch_size'] = int(config['batch_size'])
    config['iterations'] = int(config['iterations'])
    config['epsilon'] = float(config['epsilon'])
    config['eta'] = float(config['eta'])
    config['beta1'] = float(config['beta1'])
    config['beta2'] = float(config['beta2'])
    config['a_func'] = config['a_func'][0].casefold()
    config['optimizer'] = config['optimizer'][0]
    return config


def validate_config(config):
    errors = []
    n_clusters = config['counts'].shape[0]
    if config['means'].shape[0] != n_clusters or config['variances'].shape[0] != n_clusters:
        errors.append(
            f"Count of clusters differ in mean, count and variance field - {n_clusters}, {config['means'].shape[0]}, "
            f"{config['variances'].shape[0]}.")
    cluster_dimensionality = config['means'].shape[1]
    if config['variances'].shape[1] != cluster_dimensionality or config['variances'].shape[2] != cluster_dimensionality:
        errors.append(
            f"Clusters differ in mean, and variance field - {cluster_dimensionality}, {config['variances'].shape[1:]}.")
    if len(config['layers']) < 3:
        errors.append(
            f"Ensure to have at least 3 layers.")
    if config['layer'] >= len(config['layers']):
        errors.append(
            f"Layer index out of range.")
    elif config['layers'][config['layer']] != 2:
        errors.append(
            f"Selected layer does not have specified dimensionality (2).")
    if config['layers'][0] != config['layers'][-1]:
        errors.append(
            f"Input and output layer dimensionality differs.")
    for index, layer in enumerate(config['layers']):
        if layer < 1:
            errors.append(
                f"Layer {index} has invalid dimensionality - {layer}.")
    for key in ['layer', 'batch_size', 'iterations', 'epsilon', 'beta1', 'beta2', 'eta']:
        if config[key] < 0:
            errors.append(
                f"Invalid option for {key} - {config[key]}.")
    return errors
