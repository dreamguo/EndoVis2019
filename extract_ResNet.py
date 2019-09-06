import os
import cv2
import pdb
import torch
import argparse
import numpy as np
from config import *
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms


def center_data(data):
    return [np.array(data[16:240, 58:282, :])]


def oversample_data(data):
    data_flip = np.array(data[:, ::-1, :])
    
    data_1 = np.array(data[16:240, 58:282, :])
    data_2 = np.array(data[:244, :224, :])
    data_3 = np.array(data[:224, -224:, :])
    data_4 = np.array(data[-224:, :224, :])
    data_5 = np.array(data[-224:, -224:, :])

    data_f_1 = np.array(data_flip[16:240, 58:282, :])
    data_f_2 = np.array(data_flip[:224, :224, :])
    data_f_3 = np.array(data_flip[:224, -224:, :])
    data_f_4 = np.array(data_flip[-224:, :224, :])
    data_f_5 = np.array(data_flip[-224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5, data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def extract_features(frame):
    model = list(models.resnet101(pretrained=True).children())
    model = model[:-2]
    model.append(nn.AdaptiveAvgPool2d(1))
    model = nn.Sequential(*model)
    model = model.cuda()

    frame_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406],
                #    std=[0.229, 0.224, 0.225])
                ])

    with torch.no_grad():
        frame = frame_transform(frame)
        frame = torch.unsqueeze(frame, dim=0).cuda()
        frame_feature = model(frame).view(2048)
        frame_feature = frame_feature.cpu().numpy()
    
    return frame_feature


def run(video_dir, output_dir, mode_type):
    assert(mode_type in ['oversample', 'center'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for video_name in os.listdir(video_dir):
        if not video_name.endswith('.mp4'):
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
        success, frame = cap.read()
        success = True
        count = 0
        
        all_data = []
        while success:
            if frame is None:
                break
            if count % frame_rate == 0:
                if mode_type == 'oversample':
                    frame_list = oversample_data(frame)
                else:
                    frame_list = center_data(frame)
                data_list = []
                for item in frame_list:
                    data = extract_features(item)
                    data_list.append(data)
                print('ResNet - video: {} idx: {}'.format(video_name, count // frame_rate))
            success, frame = cap.read()
            count += 1
            all_data.append(data_list)
        print(count)
        cap.release()
        
        temp = video_name.split('.')
        np.savez(os.path.join(output_dir, temp[0]), feature = np.array(all_data))


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--mode_type', type=str)

    args = parser.parse_args()
    video_dir = args.video_dir
    output_dir = args.output_dir
    mode_type = args.mode_type

    print('Resnet begin')
    run(video_dir, output_dir, mode_type)

