import numpy as np
import pandas as pd
import json
from sympy import *

from glob import glob
from tqdm import tqdm
from itertools import combinations


@dataclass
class MapData:
    points: np.ndarray
    center: np.ndarray

def get_map_info(config):
    
    sub_info = get_sub_info(config)
    site = list(sub_info.keys())[config['site_num']]
    floor = sub_info[SITE].floor[config['floor_num']]
    
    floor_info_path = f'{input_dir}/metadata/{site}/{floor}/floor_info.json'
    with open(floor_info_path) as json_file:
        json_data = json.load(json_file)

    map_width = json_data["map_info"]["width"]
    map_height = json_data["map_info"]["height"]

    geojosn_map_path = f'{input_dir}/metadata/{site}/{floor}/geojson_map.json'
    with open(geojosn_map_path) as json_file:
        geojosn_map = json.load(json_file)

    profile = np.array(geojosn_map['features'][0]['geometry']['coordinates'])[0]
    min_x = np.min(profile[:,0])
    min_y = np.min(profile[:,1])
    scale_x = map_width / np.max(profile[:,0] - min_x)
    scale_y = map_height / np.max(profile[:,1] - min_y)
    
    map_info = []
    for i, geometry in enumerate(geojosn_map['features']):
        points = np.array(geometry['geometry']['coordinates'][0])
        points[:,0] = (points[:,0] - min_x) * scale_x
        points[:,1] = (points[:,1] - min_y) * scale_y
        
        if i!=0:
            center = np.array(geometry['properties']['point'])
            center[0] = (center[0] - min_x) * scale_x
            center[1] = (center[1] - min_y) * scale_y
        else:
            center = np.zeros(2)
        
        map_info.append(MapData(points, center))
           
    return map_info


def generate_waypoint_pair(
    config,
    new_waypoint_list,
    between_waypoint_max_distance,
    waypoint_and_item_center_max_distance

):
    sub_info = get_sub_info(config)
    site = list(sub_info.keys())[config['site_num']]
    floor = sub_info[SITE].floor[config['floor_num']]
    
    waypoint_dict = get_waypoint_dict(config)[site][floor]
    wp_list = waypoint_dict.wp_list
    wp_list = np.concatenate([wp_list, new_waypoint])
    wp_pair = waypoint_dict.wp_pair
    
    map_info = get_map_info(config)[1:]
    
    new_waypoint_pair = []
    for (wp1,wp2) in tqdm(list(combinations(wp_list,2))):
        if [tuple(wp1),tuple(wp2)] in wp_pair or [tuple(wp2),tuple(wp1)] in wp_pair:
            continue
        
        dist = np.sqrt(np.sum((wp1-wp2)**2))
        if dist > between_waypoint_max_distance:
            continue

        early_stopping = False
        for i, _info in enumerate(map_info):
            _dist = np.sqrt(np.sum((wp1- _info.center)**2))
            if _dist > waypoint_and_item_center_max_distance:
                continue

            points = _info.points
            for j in range(1, len(points)):
                x, y = symbols('x, y') 
                    
                xy1 = points[j-1]
                xy2 = points[j]
                try:
                    solution = solve(
                                   [(y - wp2[1])/(wp1[1] - wp2[1]) - (x - wp2[0])/(wp1[0] - wp2[0]),
                                    (y - xy2[1])/(xy1[1] - xy2[1]) - (x - xy2[0])/(xy1[0] - xy2[0]) ], 
                                    [x, y]
                                     )
                except:
                    continue
                    
                x, y = solution[x], solution[y]
                if (xy1[0]-x)*(xy2[0]-x) < 0 and (xy1[1]-y)*(xy2[1]-y) < 0 and \
                    (wp1[0]-x)*(wp2[0]-x) < 0 and (wp1[1]-y)*(wp2[1]-y) < 0:
                    early_stopping = True
                    break

            if early_stopping:
                break
            
        if not early_stopping:
            new_waypoint_pair.append([tuple(wp1),tuple(wp2)])
    
    return new_waypoint_pair