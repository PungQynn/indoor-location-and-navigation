import numpy as np
import pandas as pd

from utils import read_data_file
from dataclasses import dataclass
from glob import glob


@dataclass
class SensorData:
    dist: list
    sensor_data: list

def get_train_sensor_data(
    config,
):
    
    dist_list = []
    sensor_data = []
    
    sub_info = get_sub_info(config)
    site = list(sub_info.keys())[config['site_num']]
    
    input_dir = config['input_dir']
    filenames = glob(f'{input_dir}/train/{site}/*/*.txt')
    for fn in tqdm(filenames):
        file_data = read_data_file(fn)
        
        acce_ = file_data.acce
        gyro_ = file_data.gyro
        ahrs_ = file_data.ahrs
        posi_ = file_data.waypoint
        
        for i in range(len(posi_)-1):
            _ts = posi_[[i, i+1], 0]
            _posi = posi_[[i, i+1], 1:3]
            
            dist_list.append(np.sqrt(np.sum((_posi[1] - _posi[0])**2)))
            
            boolen_list = (_ts[0]<=acce_[:,0]) & (acce_[:,0]<_ts[1])
            _acce = acce_[boolen_list]
            _gyro = gyro_[boolen_list]
            _ahrs = ahrs_[boolen_list]
            _sensor_data = np.concatenate([_acce[:,1:],_gyro[:,1:],_ahrs[:,1:]], axis=1 )
            
            _len = 2500 - len(_sensor_data)
            _sensor_data = np.concatenate([_sensor_data, np.zeros((_len,9))])
  
            sensor_data.append(_sensor_data)

    return SensorData(np.array(dist_list), np.array(sensor_data))
            

    def generate_waypoint_timestamp_list(min_rotation_angle= 35, min_halt_time= 1500):        

        # rotation
        orientations = 180 * np.arctan2(self.stride_datas[:,2], self.stride_datas[:,1]) / np.pi
        rotations = np.abs(np.diff(orientations))
        rotations = np.where(rotations<180, rotations, 360-rotations)
        rotations = np.concatenate([self.stride_datas[:-1,:1].astype(int), rotations.astype(int).reshape(-1,1)], axis=1)
        
        rota_ts_list = []
        for rot in rotations:
            if rot[1] >= min_rotation_angle:
                rota_ts_list.append(rot[0])
        print(f'Rotation timestamp list:\n{rota_ts_list}')
        # halt
        halt = np.diff(self.stride_datas[:,0].astype(int))
        halt = np.concatenate([self.stride_datas[:-1,:1].astype(int), halt.reshape(-1,1)], axis=1)
        
        halt_ts_list1 = []
        halt_ts_list2 = []
        for i,h in enumerate(halt[:-1]):
            if h[1] >= min_halt:
                halt_ts_list1.append(h[0])
                halt_ts_list2.append((h[0], halt[i+1,0]))
        print(f'Halt timestamp list:\n{halt_ts_list2}')    
 
        return rota_ts_list, (halt_ts_list1, halt_ts_list2)


def get_test_sensor_data(
    config,
    path_pool,，
    waypoint_timestamp_list,
    
   ):
   
    sensor_data = []
   
    input_dir = config['input_dir']
    for path in path_pool:
        fn = glob(f'{input_dir}/test/{path}.txt')
        file_data = read_data_file(fn)
        
        acce_ = file_data.acce
        gyro_ = file_data.gyro
        ahrs_ = file_data.ahrs
        
        for i in range(len(waypoint_timestamp_list)-1):
            _ts = posi_[[i, i+1], 0]
            
            boolen_list = (_ts[0]<=acce_[:,0]) & (acce_[:,0]<_ts[1])
            _acce = acce_[boolen_list]
            _gyro = gyro_[boolen_list]
            _ahrs = ahrs_[boolen_list]
            _sensor_data = np.concatenate([_acce[:,1:],_gyro[:,1:],_ahrs[:,1:]], axis=1 )
            
            _len = 2500 - len(_sensor_data)
            _sensor_data = np.concatenate([_sensor_data, np.zeros((_len, 9))])
  
            sensor_data.append(_sensor_data)
        
    return np.array(sensor_data)
