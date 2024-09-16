import numpy as np
import time
import pandas as pd
import h5py


x = np.load('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/Climate_damage_crops/Climate_damage_crops_BCC-CSM2-MR_ssp245_2010-2100.npy')

"""

in_cell_df_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/cell_zones_df.pkl'
t0 = time.time()
cell_df = pd.read_pickle(in_cell_df_path)
print('Load Pickle =', time.time() - t0)

hdf = 'd:/bbrett/cell_df.h5'
cell_df.to_hdf(hdf, key='cell_df', mode='w', format="table")

t0 = time.time()
cell_hdf = pd.read_hdf(hdf)
print('Load HDF5 =', time.time() - t0)

"""


# Save as numpy arrays
t0 = time.time()
np.save('d:/bbrett/x.npy', x)
print('Save numpy array =', time.time() - t0)

# Load from numpy arrays for comparison
t0 = time.time()
x1 = np.load('d:/bbrett/x.npy')[:, 0]
print('Load numpy array =', time.time() - t0)



# Save to HDF5 from numpy arrays
t0 = time.time()
with h5py.File('d:/bbrett/data.h5', 'w') as h5f:
    h5f.create_dataset('x', data = x)#, chunks = True)
print('Save array h5py =', time.time() - t0)

# Load HDF5 from numpy arrays
t0 = time.time()
h5f = h5py.File('d:/bbrett/data.h5', 'r')
x2 = h5f['x'][:, 0]
# x2 = h5f['x'][...]
# x2 = np.array([h5f['dataset_1'][:, i] for i in range(x.shape[1])]).T
h5f.close()
print('Load array h5py =', time.time() - t0)



# # Numpy array to pandas dataframe to HDF5
# t0 = time.time()
# dfx = pd.DataFrame(x, columns = ['yr_' + str(i) for i in range(2010, 2101)])

# with h5py.File('d:/bbrett/data2.h5', 'w') as h5f:
#     h5f.create_dataset('x', data = dfx, chunks = True)
# print('Save dataframe h5py =', time.time() - t0)

# # Numpy array to pandas dataframe to HDF5
# t0 = time.time()
# h5 = h5py.File('d:/bbrett/data2.h5', 'r')
# x2 = h5['x'][:, 9]
# h5.close()
# print('Load dataframe h5py =', time.time() - t0)



# # Save to HDF5 from numpy arrays using PyTables - slow but allows reading columns by label
# t0 = time.time()
# hdf3 = 'd:/bbrett/data3.h5'
# dfx.to_hdf(hdf3, key = 'dfx', mode = 'w', format="table") # , data_columns = True)
# print('PyTables to_hdf =', time.time() - t0)

# t0 = time.time()
# dfxx = pd.read_hdf(hdf3, key = 'dfx', columns = ['yr_2010'])
# print('PyTables read_hdf =', time.time() - t0)


