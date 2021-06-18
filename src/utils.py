import sys
import numpy as np
import pandas as pd
import multiprocessing
import pickle

from glob import glob
from tqdm import tqdm
from dataclasses import dataclass

import json
import plotly.graph_objs as go
from PIL import Image


@dataclass
class SubData:
    path: np.ndarray
    floor: np.ndarray

def get_sub_info(config):
    input_dir = config['input_dir']
    
    sub = pd.read_csv(f'{input_dir}/sample_submission.csv')
    sub['site'] = sub.site_path_timestamp.apply(lambda x: (x.split("_")[0]))
    sub['path'] = sub.site_path_timestamp.apply(lambda x: (x.split("_")[1]))
    
    sub_info = {}    
    for site in np.unique(sub.site):
        sub_info[site] = {}
        # floor list
        filenames = glob(f'{input_dir}/train/{site}/*/*.txt')
        floors = []
        for fn in filenames:
            floors.append(fn.split("/")[-2])
        
        sub_info[site] = SubData(np.unique(sub[sub.site==site].path), np.unique(floors))
        
    return sub_info


def get_floor_dict():
    floor_str = ["B2", "B1",
                 "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                 "1F", "2F", "3F", "4F", "5F", "6F", "7F", "8F", "9F"]
    floor_num = [-2,-1] + 2*list(range(0,9))

    floor_dict = dict(zip(floor_str, floor_num))
    for i, num in enumerate(floor_num):
        if num in floor_dict:
            floor_dict[num].append(floor_str[i])
        else:
            floor_dict[num] = [floor_str[i]]
    
    return floor_dict


def get_path_time(config):
    input_dir = config['input_dir']
    try:
        fn = glob(f'{input_dir}/train/*/*/{path}.txt')[0]
        with open(fn, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        startTime = int(lines[0].split('\t')[1][10:])
        endTime = int(lines[-1].split('\t')[1][8:])                
    except:
        fn = glob(f'{input_dir}/test/{path}.txt')[0]
        with open(fn, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        startTime = int(lines[0].split('\t')[2])
        endTime = int(lines[-1].split('\t')[2])
        
    return (startTime, endTime)


@dataclass  # learn this
class ReadData:
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    waypoint: np.ndarray

def read_data_file(data_filename):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    ahrs = []
    wifi = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ROTATION_VECTOR':
            ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts = line_data[0]
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            lastseen_ts = line_data[6]
            wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    waypoint = np.array(waypoint)

    return ReadData(acce, acce_uncali, gyro, gyro_uncali, ahrs, wifi, waypoint)


def get_train_info(config):
    input_dir = config['input_dir']
    private_input_dir = config['private_input_dir']
    
    try:
        train_info = pd.read_csv(f'{private_input_dir}/data/train_info.csv')
        
    except:
        values = np.zeros((0,9))
        
        floor_dict = get_floor_dict()
        for site in tqdm(get_sub_info(config).keys()):
            filenames = glob(f'{input_dir}/train/{site}/*/*.txt')
            for fn in filenames:
                floor = fn.split('/')[-2]
                floor_num = floor_dict[floor]
                path = fn.split('/')[-1][:-4]
                time = get_path_time(path)
                waypoints = read_data_file(fn).waypoint[:,1:3]
                values = np.concatenate([
                         values, 
                         np.array([[site,path,floor_num,time[0],time[1],waypoints[0,0],waypoints[0,1],waypoints[-1,0],waypoints[-1,1]]])
                                        ])
            
        train_info = pd.DataFrame(values, columns= ['site','path','floor','startTime','endTime','startWp_x','startWp_y','endWp_x','endWp_y'])
        
        train_info.to_csv("train_info.csv", index=False)
        
    return train_info


@dataclass
class WaypointData:
    wp_list: np.ndarray
    wp_pair: np.ndarray

def get_waypoint_dict(config): 
    input_dir = config['input_dir']
    private_input_dir = config['private_input_dir']
    
    try:
        with open(f"{private_input_dir}/data/wp_dict.pkl", 'rb') as f:
            wp_dict = pickle.load(f)        
    except:
        wp_dict = {}    
        sub_info = get_sub_info(config)
        
        for site in tqdm(sub_info.keys()):
            wp_dict[site] = {}
            for floor in sub_info[site].floor:
                _wp_list, _wp_pair_list = [], []
                
                filenames = glob(f'{input_dir}/train/{site}/{floor}/*.txt')
                for fn in (filenames):
                    datas = read_data_file(fn)
                    for i, waypoint in enumerate(datas.waypoint):
                        wp = tuple(waypoint[1:3])
                        if wp not in _wp_list:
                            _wp_list.append(wp)

                        if i !=0:
                            _wp = tuple(datas.waypoint[i-1,1:3])
                            _wp_pair = [_wp, wp]
                            if _wp_pair not in _wp_pair_list:
                                _wp_pair_list.append(_wp_pair)
                
                wp_dict[site][floor] = WaypointData(_wp_list, _wp_pair_list)
                
        with open("wp_dict.pkl", 'wb') as f:
            pickle.dump(wp_dict, f)
        
    return wp_dict


def visualize_trajectory(
    trajectory, 
    config, 
    title=None, 
    mode='lines + markers + text', 
    show=False
):
    
    sub_info = get_sub_info(config)
    site = list(sub_info.keys())[config['site_num']]
    floor = sub_info[SITE].floor[config['floor_num']]
    
    input_dir = config['input_dir']
    json_plan_filename = f'{input_dir}/metadata/{site}/{floor}/floor_info.json'
    with open(json_plan_filename) as json_file:
        json_data = json.load(json_file)
    
    map_width = json_data["map_info"]["width"]
    map_height = json_data["map_info"]["height"]
    
    fig = go.Figure()

    # add trajectory
    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    color_list = ['rgba(4, 174, 4, 0.5)'] * trajectory.shape[0]
    color_list[0] = 'rgba(12, 5, 235, 1)'
    color_list[-1] = 'rgba(235, 5, 5, 1)'

    position_count = {}
    text_list = []
    for i in range(trajectory.shape[0]):
        if str(trajectory[i]) in position_count:
            position_count[str(trajectory[i])] += 1
        else:
            position_count[str(trajectory[i])] = 0
        text_list.append('        ' * position_count[str(trajectory[i])] + f'{i}')
    text_list[0] = 'Start Point: 0'
    text_list[-1] = f'End Point: {trajectory.shape[0] - 1}'

    fig.add_trace(
        go.Scattergl(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode=mode,
            marker=dict(size=size_list, color=color_list),
            line=dict(shape='linear', color='rgb(100, 10, 100)', width=2, dash='dot'),
            text=text_list,
            textposition="top center",
            name='trajectory',
        ))

    # add floor plan
    floor_plan = Image.open(f'{input_dir}/metadata/{site}/{floor}/floor_image.png')
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y= map_height,
            sizex= map_width,
            sizey= map_height,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, map_width])
    fig.update_yaxes(autorange=False, range=[0, map_height], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=  900,
        height= 200 + 900 * map_height / map_width,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig