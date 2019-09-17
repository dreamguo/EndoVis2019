from config import *
import os
import ipdb
import random
import argparse
import numpy as np

def normalize_data(norm_v, global_mean, global_std): 
    #ipdb.set_trace()
    #norm_v = data[global_std > feature_std_thrh, :]    # 不用删除特征了
    #norm_v = norm_v - global_mean[global_std > feature_std_thrh]
    #norm_v = norm_v / global_std[global_std > feature_std_thrh]
    
    norm_v = norm_v - global_mean
    norm_v = norm_v / global_std
    
    return norm_v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    
    args = parser.parse_args()
    feature_dir = args.feature_dir
    output_dir = args.output_dir
    
    name_list = os.listdir(feature_dir)
    data_short = []
    for name_item in name_list:
        if not name_item.endswith('.npz'):
            name_list.remove(name_item)
            continue
        print('Calculate mean and std: ' + name_item)
        feature_npz = np.load(os.path.join(feature_dir, name_item))
        feature = feature_npz['feature'].tolist()
        feature_short = []
        for i in range(0, len(feature)):
            feature_tmp = np.array(feature[i])[::4, :]
            feature_short.append(np.transpose(feature_tmp))
        data_short.append(np.concatenate(feature_short, axis=1))
    data_short = np.concatenate(data_short, axis=1)
    global_mean = data_short.mean(1)
    global_std = data_short.std(1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    length = 0
    for name_item in name_list:
        feature_npz = np.load(os.path.join(feature_dir, name_item))
        feature = feature_npz['feature'].tolist()
        data = []
        for i in range(0, len(feature)):                            # 10 or 1 * frames * 1024
            feature_tmp = np.array(feature[i])                      # frames * 1024
            data_new = normalize_data(feature_tmp, global_mean, global_std)
            data.append(data_new)
        # 10 or 1 * frames * xxxx( <= 1024)
        print('Replace data: {} from {} to {}'.format(name_item, np.array(feature).shape, np.array(data).shape))
        
        #assert(len(data[0]) * i3d_time == feature_npz['frame_cnt']) # i3d中,frame_cnt略大于i3d_time*len(data[0])
        np.savez(os.path.join(output_dir, name_item), 
                feature = np.array(data), 
                frame_cnt = feature_npz['frame_cnt'], 
                video_name = name_item)

