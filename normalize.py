from config import *
import os
import ipdb
import random
import argparse
import numpy as np

def normalize_data(data):
    global_mean = data.mean(1)
    global_std = data.std(1)
    
    #ipdb.set_trace()
    norm_v = data[global_std > feature_std_thrh, :]
    global_std = np.transpose(global_std)
    global_mean = np.transpose(global_mean)
    norm_v = np.transpose(norm_v)

    norm_v = norm_v - global_mean[global_std > feature_std_thrh]
    norm_v = norm_v / global_std[global_std > feature_std_thrh]
    norm_v = np.transpose(norm_v)

    return norm_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    
    args = parser.parse_args()
    feature_dir = args.feature_dir
    output_dir = args.output_dir
    
    name_list = os.listdir(feature_dir)
    video_len = []
    data_tmp = []
    for name_item in name_list:
        print(name_item)
        if not name_item.endswith('.npz'):
            name_list.remove(name_item)
            continue
        feature_npz = np.load(os.path.join(feature_dir, name_item))
        feature = feature_npz['feature'].tolist()
        i3d_types = len(feature)
        video_len.append(len(feature[0]))
        tmp = []
        for i in range(0, len(feature)):
            tmp.append(np.transpose(feature[i]))
        data_tmp.append(np.concatenate(tmp, axis=1))
    data = np.concatenate(data_tmp, axis=1)
    data_new = normalize_data(data)
    print(data.shape, end=' new: ')
    print(data_new.shape)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    length = 0
    for i in range(len(video_len)):
        tmp_list = []
        for j in range(i3d_types):
            tmp_list.append(np.transpose(data_new[:, length : length + video_len[i]]))
            length += video_len[i]
        np.savez(os.path.join(output_dir, name_list[i]), feature = np.array(tmp_list))


