import my_utils as ut
import numpy as np

coords, ages, info_finnished_arr = ut.get_coords_and_ages('extracted_sampleInfo.txt','ancient_bamfilelist.txt')

data = []
labels = []
for i in range(200000):
    input_, idx, N, start_pos, chro = ut.test('glf_database.h5', 'mafs_data.h5')
    data.append((input_, start_pos, chro))
    labels.append(info_finnished_arr[int(N[4:])-1])

labels_arr = np.array(labels)
data_arr = np.array(data)

np.save('labels_200k.npy', labels_arr)
np.save('data_200k.npy', data_arr)

