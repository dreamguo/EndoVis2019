from config import *

def normalize_data(data):

    temp_cat = []
    for k, v in data.items():
        temp_cat.append(v)
    temp_cat = np.concatenate(temp_cat, axis=0)

    global_mean = temp_cat.mean(0)
    global_std = temp_cat.std(0)

    for k, v in data.items():

        norm_v = v[:,global_std > feature_std_thrh] 
        norm_v = norm_v - global_mean[global_std > feature_std_thrh]
        norm_v = norm_v / global_std[global_std > feature_std_thrh]

        data[k] = norm_v

    return data

def main():

main()
