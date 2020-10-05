import pysam
import h5py
import numpy as np
import timeit
import random
import pandas as pd
import operator
import matplotlib.pyplot as plt


def resize(resize_dim, pos_, gl, region_size):
    
    m = np.zeros((resize_dim, 3))
    for i, pos in enumerate(pos_):
        j = int(pos * resize_dim / region_size)
        m[j, :] += gl[i]
    
    return m



def parameters():
    
    seg_sites = [2606694,1551724,983510,958052,905056,736677,747172,454489,447438,1509902,1548781,
                 1516390,1265635,1255756,1172649,1034231,618102,885768,660241,518882,310259,330861]
    
    seg_sites_arr = np.array(seg_sites)-150000
    
    idx = random.randint(0, 21)
    
    rand_start = random.randint(1, seg_sites_arr[idx])
    rand_end = rand_start + 150000
    
    chromosome = 'chr_'+str(idx+1)
    
    indi = 'ind_'+str(random.randint(1, 147))
    
    return rand_start, rand_end, chromosome, indi


def region(glf_data):
    
    start_pos = glf_data[0][0]
    end_pos = start_pos + 4e6
    idx = []
    for i, val in enumerate(glf_data):
        if val[0] < end_pos:
            continue
        else:
            idx.append(i)
            break
    
    return idx[0], start_pos



def major_minor(h, idx, start_pos, mafs):
    
    gl_mat = []
    genomic_pos = []
    for i, val in enumerate(h[0:idx]):
        if mafs[i][2] == b'A' and mafs[i][3] == b'C' or mafs[i][2] == b'C' and mafs[i][3] == b'A':
            gl_mat.append(np.array(operator.itemgetter(1,2,5)(val)))
            genomic_pos.append(h[i][0] - start_pos)
        elif mafs[i][2] == b'A' and mafs[i][3] == b'G' or mafs[i][2] == b'G' and mafs[i][3] == b'A':
            gl_mat.append(np.array(operator.itemgetter(1,3,8)(val)))
            genomic_pos.append(h[i][0] - start_pos)
        elif mafs[i][2] == b'A' and mafs[i][3] == b'T' or mafs[i][2] == b'T' and mafs[i][3] == b'A':
            gl_mat.append(np.array(operator.itemgetter(1,4,10)(val)))
            genomic_pos.append(h[i][0] - start_pos)
        elif mafs[i][2] == b'C' and mafs[i][3] == b'G' or mafs[i][2] == b'G' and mafs[i][3] == b'C':
            gl_mat.append(np.array(operator.itemgetter(5,6,8)(val)))
            genomic_pos.append(h[i][0] - start_pos)
        elif mafs[i][2] == b'C' and mafs[i][3] == b'T' or mafs[i][2] == b'T' and mafs[i][3] == b'C':
            gl_mat.append(np.array(operator.itemgetter(5,7,10)(val)))
            genomic_pos.append(h[i][0] - start_pos)
        elif mafs[i][2] == b'G' and mafs[i][3] == b'T' or mafs[i][2] == b'T' and mafs[i][3] == b'G':
            gl_mat.append(np.array(operator.itemgetter(8,9,10)(val)))
            genomic_pos.append(h[i][0] - start_pos)
        else:
            print('ERROR')
    
    return gl_mat, genomic_pos



def test(glf_file, mafs_file):
    
    with h5py.File(glf_file, 'r') as f, h5py.File(mafs_file, 'r') as f1:

        start, end, chro, N = parameters()

        mafs = f1['Chromosomes/'+chro][start:end]

        h = f[chro+'/'+N][start:end]

        idx, start_pos = region(h)

        if idx < 40:
            idx, start_pos = region(h)

        gl_mat, genomic_pos = major_minor(h, idx, start_pos, mafs)

        input_ = resize(128, genomic_pos, gl_mat, 4e6)
        
        return input_, idx, N, start_pos, chro


def get_coords_and_ages(infofile, bamfile):

    with open(infofile, 'r') as f:
        info = []
        for i in f.readlines():
            info.append(i.split('\t'))

    with open(bamfile, 'r') as f:
        bam_file = []
        for i in f.readlines():
            bam_file.append(i.split('\t'))

    sorted_info = []
    for i in bam_file:
        for j in info:
            if j[0] in i[0]:
                sorted_info.append(j)

    info_finnished = []
    idx = [113, 115, 116, 118]
    for i, val in enumerate(sorted_info):
        if i not in idx:
            info_finnished.append(val)

    info_finnished_arr = np.array(info_finnished)
    coords = np.float_(info_finnished_arr[:, 6:8])
    ages = np.float_(info_finnished_arr[:, 9])

    return coords, ages, info_finnished_arr

#coords, ages = get_coords_and_ages('extracted_sampleInfo.txt','ancient_bamfilelist.txt')


