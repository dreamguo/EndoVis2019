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


def extract_features(frame):

    model = list(models.resnet101(pretrained=True).children())
    model = model[:-2]
    model.append(nn.AdaptiveAvgPool2d(1))
    model = nn.Sequential(*model)
    model = model.cuda()

    frame_transform = transforms.Compose([
                #transforms.Resize((270, 480)),  # (540, 960, 3)
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

    with torch.no_grad():
        frame = frame_transform(frame)
        frame = torch.unsqueeze(frame, dim=0).cuda()
        frame_feature = model(frame).view(1, 2048)
        frame_feature = frame_feature.cpu().numpy()
    
    return frame_feature


def run(video_dir, output_dir):
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
                data = extract_features(frame)
                print('ResNet - video {} idx {}'.format(video_name, count // frame_rate))
            success, frame = cap.read()
            count += 1
            all_data.append(data)
        print(count)
        cap.release()
        
        temp = video_name.split('.')
        np.savez(os.path.join(output_dir, temp[0]), feature = np.array(all_data))


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    video_dir = args.video_dir
    output_dir = args.output_dir
    print('Resnet begin')
    run(video_dir, output_dir)

