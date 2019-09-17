import os
import cv2
import ipdb
import torch
import random
import argparse
import numpy as np
#from config import *
import torch.nn as nn
from torchvision import models
from torchvision import transforms


def center_data(data):
    return [data[16:240, 58:282, :]]


def oversample_data(data):
    data_flip = data[:, ::-1, :]
    
    data_1 = data[16:240, 58:282, :]
    data_2 = data[:224, :224, :]
    data_3 = data[:224, -224:, :]
    data_4 = data[-224:, :224, :]
    data_5 = data[-224:, -224:, :]

    data_f_1 = data_flip[16:240, 58:282, :]
    data_f_2 = data_flip[:224, :224, :]
    data_f_3 = data_flip[:224, -224:, :]
    data_f_4 = data_flip[-224:, :224, :]
    data_f_5 = data_flip[-224:, -224:, :]

    return [data_1, data_2, data_3, data_4, data_5, data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def extract_features(frame, model):
    frame_transform = transforms.Compose([
                #transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                ])

    with torch.no_grad():
        frame = np.transpose(frame, (2, 0, 1))              # ToTensor change 224*224*3 into 3*224*224
        frame = np.array(frame, dtype = np.float32)         # ToTensor change np.int8 into np.float32
        frame = torch.from_numpy(frame)                     # ToTensor change numpy/Img into Tensor
        
        frame = frame_transform(frame)
        frame = torch.unsqueeze(frame, dim=0).cuda().float()
        frame_feature = model(frame).view(2048)
        frame_feature = frame_feature.cpu().numpy()
    
    return frame_feature


def run(video_name, data, model):
    #print(np.array(data).shape) 
    
    feature = []
    for i in range(len(data)):                              # i in batch_size
        data_list = []
        for j in range(len(data[i])):                       # j in 10 or 1
            data_out = extract_features(data[i][j], model)  # 2048 or last_part(<2048)
            data_list.append(data_out)
        feature.append(data_list)                           # 10 or 1 * 2048 or last_part

    return feature                                          # batch_size * 10 or 1 * 2048 or last_part


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--mode_type', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--frame_rate', type=int)

    args = parser.parse_args()
    video_dir = args.video_dir
    output_dir = args.output_dir
    mode_type = args.mode_type
    batch_size = args.batch_size
    frame_rate = args.frame_rate
    
    print('=' * 10 + 'Resnet begin' + '=' * 10) 
    assert(mode_type in ['oversample', 'center'])
    assert(frame_rate >= 1)
    
    model = list(models.resnet101(pretrained=True).children())
    model = model[:-2]
    model.append(nn.AdaptiveAvgPool2d(1))
    model = nn.Sequential(*model)
    model = model.cuda()
    
    jump_list = [
            #'Hei-Chole11.mp4',
            ]
    for video_name in os.listdir(video_dir):
        if not video_name.endswith('.mp4'):
            print('Jump: ' + video_name)
            continue
        if video_name in jump_list:
            print('Jump: ' + video_name)
            continue
        
        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
        success, frame = cap.read()
        success = True
        count = 0 
        frame_cnt = 0;
        
        feature = []
        batch_data = []
        while success:
            if frame is None:
                break
            if count % frame_rate == 0:
                if mode_type == 'oversample':
                    batch_data.append(oversample_data(frame))
                else:
                    batch_data.append(center_data(frame))
                frame_cnt += 1
            if len(batch_data) == batch_size:
                print('{}/{}: idex:{}'.format(video_dir, video_name, frame_cnt - batch_size))
                feature.append(run(video_name, batch_data, model))
                batch_data = []
            success, frame = cap.read()
            count += 1
        if (len(batch_data) != 0):                          # last_part
            print('{}/{}: idex:{}'.format(video_dir, video_name, frame_cnt - len(batch_data)))
            feature.append(run(video_name, batch_data, model))
        cap.release()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        feature = np.concatenate(feature, axis=0)
        feature = np.transpose(np.array(feature), (1, 0, 2))
        print('{}/{}: output{}'.format(video_dir, video_name, feature.shape))  # 10 or 1 * frames * 2048
        temp = video_name.split('.')
        np.savez(os.path.join(output_dir, temp[0]),
                feature = feature,
                frame_cnt = frame_cnt,
                video_name = temp)    

