import os
import torch
import numpy as np
import random
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess
from config import *
import ipdb


def get_test_combination_cases(feature_name_list, feature_type, length):
    rgb_list, flow_list = [], []
    for i in range(len(feature_name_list)):
        tmp = feature_name_list[i].split('-')
        tmp = tmp[:-1]
        tmp = '-'.join(tmp)
        rgb_list.append(tmp + '-rgb.npz')
        flow_list.append(tmp + '-flow.npz')
    
    if (len(feature_type) < 5):
        cases_rgb, flips_nums, frame_nums = get_test_cases(rgb_list, 'rgb', length)
        cases_flow, flips_nums, frame_nums = get_test_cases(flow_list, 'flow', length)
    elif feature_type.endswith("oversample_4"):
        cases_rgb, flips_nums, frame_nums = get_test_cases(rgb_list, 'rgb_oversample_4', length)
        cases_flow, flips_nums, frame_nums = get_test_cases(flow_list, 'flow_oversample_4', length)
    elif feature_type.endswith("oversample_4_norm"):
        cases_rgb, flips_nums, frame_nums = get_test_cases(rgb_list, 'rgb_oversample_4_norm', length)
        cases_flow, flips_nums, frame_nums = get_test_cases(flow_list, 'flow_oversample_4_norm', length)
    for i in range(len(cases_rgb)):
        flips_nums[i] = flips_flows[i]
        for j in range(len(cases_rgb[i])):
            for n in range(len(cases_flow[i][j])):
                cases_rgb[i][j].append(cases_flow[i][j][n])
    assert(len(cases_rgb) = len(cases_flows))
    assert(len(cases_rgb[0]) = len(cases_flows[0]))
    assert(len(cases_rgb[0][0]) > len(cases_flows[0][0]))
    assert(len(cases_rgb[0][0]) == input_dim)
    return cases_rgb, flips_nums, frame_nums


def get_test_cases(feature_name_list, feature_type, length):
    if new_data == 'True':
        feature_dir = '../i3d_new'
    else:
        feature_dir = '../i3d'
    feature_dir = os.path.join(feature_dir, feature_type)
    test_cases = []
    flips_nums = []
    frame_nums = []
    for name_item in feature_name_list:
        feature_npz = np.load(os.path.join(feature_dir, name_item))
        print(feature_dir + '/' + name_item, ' for test')
        feature = feature_npz['feature'].tolist()
        idx = random.randint(0, len(feature) - 1)
        feature = feature[idx]
        if (len(feature) < length):
            print('video length is ', len(feature))
            print('video is shorter than ', length)
            exit()
        else:
            flips_num = len(feature) // length
            flips_nums.append(flips_num)
            frame_nums.append(len(feature))
        flips_feature = []
        for flips_id in range(flips_num):
            flips_feature.append(feature[flips_id * length : (flips_id + 1) * length])
        test_cases.append(flips_feature)
    return test_cases, flips_nums, frame_nums


def get_train_combination_case(feature_name_list, feature_type, length):
    rgb_list, flow_list = [], []
    for i in range(len(feature_name_list)):
        tmp = feature_name_list[i].split('-')
        tmp = tmp[:-1]
        tmp = '-'.join(tmp)
        rgb_list.append(tmp + '-rgb.npz')
        flow_list.append(tmp + '-flow.npz')
    idx = random.randint(0, len(feature_name_list) - 1)

    if (len(feature_type) < 5):
        data_rgb, name, frame, frame_num = get_train_case(rgb_list, 'rgb', length, rgb_list[idx])
        data_flow, name, frame, frame_num = get_train_case(flow_list, 'flow', length, flow_list[idx], frame)
    elif (feature_type.endswith("oversample_4")):
        data_rgb, name, frame, frame_num = get_train_case(rgb_list, 'rgb_oversample_4', length, rgb_list[idx])
        data_flow, name, frame, frame_num = get_train_case(flow_list, 'flow_oversample_4', length, flow_list[idx], frame)
    elif (feature_type.endswith("oversample_4_norm")):
        data_rgb, name, frame, frame_num = get_train_case(rgb_list, 'rgb_oversample_4_norm', length, rgb_list[idx])
        data_flow, name, frame, frame_num = get_train_case(flow_list, 'flow_oversample_4_norm', length, flow_list[idx], frame)
    #print(data_rgb.shape)
    #print(data_flow.shape)
    data = np.concatenate((data_rgb, data_flow), axis=1)
    #print(data.shape)
    return data, name, frame, frame_num


def get_train_case(feature_name_list, feature_type, length, name='', start_frame=-1):
    if name == '':
        name = feature_name_list[random.randint(0, len(feature_name_list) - 1)]
    if new_data == 'True':
        feature_dir = '../i3d_new'
    else:
        feature_dir = '../i3d'
    feature_dir = os.path.join(feature_dir, feature_type)
    feature_npz = np.load(os.path.join(feature_dir, name))
    print(feature_dir + '/' + name, ' for train')
    feature = list(feature_npz['feature'])
    idx = random.randint(0, len(feature) - 1)
    feature = feature[idx]
    frame_num = len(feature)
    if (len(feature) < length):
        print('video length is ', len(feature))
        print('video is shorter than ', length)
        exit()
    else:
        if start_frame == -1:
            start_frame = random.randint(0, len(feature) - length)
    data = feature[start_frame : (start_frame + length)]
    data = np.array(data)
    return data, name, start_frame, frame_num


def get_test_gt(feature_name, flips_num, length, frame_num):
    gts = []
    for i in range(flips_num):
        gt = {}
        gt['gt_phase'], gt['gt_instrument'], gt['gt_action'], gt['gt_action_detailed'], gt['gt_calot_skill'], gt['gt_dissection_skill'] = get_train_gt(feature_name, i * length, length, frame_num)
        gts.append(gt)
    return gts


def get_train_gt(feature_name, feature_frame, length, feature_frame_num):
    gt_frame = feature_frame * i3d_time
    gt_frame_num = feature_frame_num * i3d_time
    flag = 1

    tmp = feature_name.split('-')
    name = '-'.join([tmp[0], tmp[1]]) + '_'
    # print("in get_train_gt: ", name)
    if new_data == 'True':
        gt_dir = '../New_Annotations/Action'
        gt_paths_1 = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if (i.endswith('.csv') and (i.startswith('Hei-C' + name[5:]) or i.startswith('hei-c' + name[5:])))]
        gt_dir = '../New_Annotations/Phase'
        gt_paths_2 = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if (i.endswith('.csv') and (i.startswith('Hei-C' + name[5:]) or i.startswith('hei-c' + name[5:])))]
        gt_dir = '../New_Annotations/Instrument'
        gt_paths_3 = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if (i.endswith('.csv') and (i.startswith('Hei-C' + name[5:]) or i.startswith('hei-c' + name[5:])))]
        gt_dir = '../New_Annotations/Skill'
        gt_paths_4 = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if (i.endswith('.csv') and (i.startswith('Hei-C' + name[5:]) or i.startswith('hei-c' + name[5:])))]
        gt_paths = gt_paths_1 + gt_paths_2 + gt_paths_3 #+ gt_paths_4
    else:
        gt_dir = '../Annotations/'
        gt_paths = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if (i.endswith('.csv') and (i.startswith('Hei-C' + name[5:]) or i.startswith('hei-c' + name[5:])))]
    #ipdb.set_trace()
    for gt_path in gt_paths:
        #print(gt_path)
        tmp1 = np.loadtxt(gt_path, delimiter=",")
        tmp1 = np.array(tmp1)
        #print(tmp1.shape)
        gt_data = tmp1[0:, 1:]
        if flag == 1:
            flag = 0
            gt_time = round(gt_data.shape[0] / gt_frame_num)
    
        gt_data = gt_data[gt_frame * gt_time : (gt_frame + i3d_time * length) * gt_time, 0:]
        if gt_time > 1: # if gt_frames is more than feature_frames
            #print(gt_path)
            #print('gt_time: ' + str(gt_time))
            tmp_frame = 0
            data = []
            while tmp_frame < i3d_time * length * gt_time:
                data.append(gt_data[tmp_frame, 0:])
                tmp_frame += gt_time
            gt_data = np.array(data)

        # without skill, must have the sentence below
        gt_calot_skill = []
        gt_dissection_skill = []

        if (gt_path.endswith('hase.csv')):
            # print('phase ', end='')
            # print(gt_data.shape)
            gt_phase = gt_data
        elif (gt_path.endswith('nstrument.csv')):
            # print('instrument ', end='')
            # print(gt_data.shape)
            gt_instrument = gt_data
        elif (gt_path.endswith('ction.csv')):
            # print('action ', end='')
            # print(gt_data.shape)
            gt_action = gt_data
        elif (gt_path.endswith('etailed.csv')):
            # print('action_detailed ', end='')
            # print(gt_data.shape)
            gt_action_detailed = gt_data
        #elif (gt_path.endswith('calot_skill.csv')):
            #gt_calot_skill = gt_data
        #elif (gt_path.endswith('dissection_skill.csv')):
            #gt_dissection_skill = gt_data
    return gt_phase, gt_instrument, gt_action, gt_action_detailed, gt_calot_skill, gt_dissection_skill


def get_phase_error(pred_phase, gt_phase):
    # pred_phase: numpy, frames x 7
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred_phase, gt_phase.squeeze())
    return loss


def get_instrument_error(pred_instrument, gt_instrument):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred_instrument, gt_instrument)
    return loss


def get_action_error(pred_action, gt_action):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred_action, gt_action)
    return loss


def get_acc_phase(pred_phase, gt_phase):
    assert(len(pred_phase) == len(gt_phase))
    right = 0
    wrong = 0
    for i in range(len(gt_phase)):
        #print('pred: ', end='')
        #print(pred_phase[i].cpu().numpy())
        #print(np.argmax(pred_phase[i].cpu().numpy(), axis=0))
        #print('gt: ', end='')
        #print(gt_phase[i].cpu().numpy())
        if (np.argmax(pred_phase[i].cpu().numpy(), axis=0) == gt_phase[i].cpu().numpy()):
            right += 1
        else:
            wrong += 1
    #print('len_all: ' + str(len(pred_phase)))
    #print('right' + str(right))
    #print('wrong' + str(wrong))
    return right, wrong


def num2name(num_list, feature_type):
    name_list = []
    for item in num_list:
        s = 'Hei-Chole' + item
        if feature_type.startswith('flow'):
            feature_type = 'flow'
        elif feature_type.startswith('rgb'):
            feature_type = 'rgb'
        else:
            print('There is something wrong with the feature_type!')
        s = s + '-' + feature_type + '.npz'
        name_list.append(s)
    return name_list
