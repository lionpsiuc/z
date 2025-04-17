import pathlib
import pandas as pd
import numpy as np

datafile = pathlib.Path('timing_data.csv')
datafile_double = pathlib.Path('timing_data_double.csv')
full_data = pd.concat([pd.read_csv(datafile), pd.read_csv(datafile_double)])

cpu_data = full_data.loc[full_data['skip_cpu'] == 0]
cpu_data = cpu_data.groupby(['height', 'width', 'precision'], as_index=False).median()
cpu_data = cpu_data[['height', 'width', 'precision', 'cpu_iteration', 'cpu_average']]

data = full_data.loc[full_data['skip_cpu'] != 0]
data = data.groupby(
    ['block_size', 'height', 'width', 'precision', 'device_index'], as_index=False).median()
data = data.drop('cpu_iteration', axis=1)
data = data.drop('cpu_average', axis=1)


data = data.merge(cpu_data, left_on=['height', 'width', 'precision'], right_on=[
    'height', 'width', 'precision'], suffixes=(False, False))
data = data[['block_size', 'height', 'width', 'precision', 'device_index',
             'cpu_iteration', 'gpu_iteration', 'gpu_iteration_comp',
             'cpu_average', 'gpu_average', 'gpu_average_comp']]

data['iter_speedup_total'] = data.cpu_iteration / data.gpu_iteration
data['iter_speedup_comp'] = data.cpu_iteration / data.gpu_iteration_comp

data['avg_speedup_total'] = data.cpu_average / data.gpu_average
data['avg_speedup_comp'] = data.cpu_average / data.gpu_average_comp

iter_data = data.loc[((data.device_index == 0) & (data.block_size == 256)) | ((data.device_index == 1) & (data.block_size == 992))]
iter_data = iter_data.sort_values(by=['device_index', 'height', 'precision'])

print("Iteration stats")
print(iter_data[['device_index', 'precision', 'height', 'gpu_iteration_comp', 'iter_speedup_comp', 'iter_speedup_total']].round(4))

avg_data = data.loc[((data.device_index == 0) & (data.block_size == 32)) | ((data.device_index == 1) & (data.block_size == 32))]
avg_data = avg_data.sort_values(by=['device_index', 'height', 'precision'])

print("Average stats")
print(avg_data[['device_index', 'precision', 'height', 'gpu_average_comp', 'avg_speedup_comp', 'avg_speedup_total']].round(4))

