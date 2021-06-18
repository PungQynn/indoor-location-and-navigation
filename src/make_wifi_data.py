import numpy as np
import pandas as pd

import utils
import compute_f

from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d
from utils import read_data_file

input_dir = '../input/indoor-location-navigation'
extend_input_dir = '../input/indoorlocationandnavigation'
wifi_cols = ['ts', 'ssid', 'bssid', 'rssi', "lsts"]
wifi_dtypes = {'ts':int, 'ssid':str, 'bssid':str, 'rssi':int, 'lsts':int}

def wifi_to_positions(
    filenames,
    generate_points = True,
    grid_distance = 0.1,
    tylor_approximation = 'first',
    add_noises = False,
    mu = 0,
    sigma = 0.1,
    random_state = 42
    ):
    
    wifi_to_xy = {}
    step_point_list = np.zeros((0,3))
    wifi_point_list = np.zeros((0,2))
    
    
    for name in tqdm(filenames):
        _datas = read_data_file(name)
        acce_, ahrs_, posi_, wifi_ = _datas.acce, _datas.ahrs, _datas.waypoint, _datas.wifi
        
        step_positions = compute_f.compute_step_positions(acce_, ahrs_, posi_)
        step_point_list = np.concatenate([step_point_list, step_positions]) # update list
        
        to_ts =  interp1d(step_positions[:,0],step_positions[:,1:3], axis=0) # learn this!!
        
        tmp = np.zeros((0,2))
        sep_tss = np.unique(wifi_[:, 0].astype(float))
        wifi_list = compute_f.split_ts_seq(wifi_, sep_tss)
        for _wifi_ in wifi_list:
            ts = float(_wifi_[0,0])
            try:
                xy = to_ts(ts)
            except:
                diff = np.abs(step_positions[:,0] - ts)
                _ts = step_positions[np.argmin(diff),0]
                xy = to_ts(_ts)
            
            wifi_df = pd.DataFrame(_wifi_, columns=wifi_cols).astype(wifi_dtypes)
            wifi_df = wifi_df.groupby('bssid')['rssi'].mean().reset_index()
            wifi_to_xy[tuple(xy)] = wifi_df

            tmp = np.concatenate([tmp, xy.reshape(-1,2)]) 
            
        wifi_point_list = np.concatenate([wifi_point_list, tmp]) # update list
        
        
        if generate_points:
            
            for i in range(tmp.shape[0]-1):
                p1 = tmp[i,:]
                p2 = tmp[i+1,:]
                d = np.sqrt(np.sum((p2-p1)**2))
                
                if d>grid_distance:
                    wifi1 = wifi_to_xy[tuple(p1)]
                    wifi2 = wifi_to_xy[tuple(p2)]
                    df = pd.merge(wifi1,wifi2,on='bssid',how='left')
                    wifi_grad = df.rssi_y - df.rssi_x
                    
                    for i in range(int(d//grid_distance)+1):
                        _p = p1 + (p2-p1)*i*grid_distance/d
                        #wifi_point_list = np.concatenate([wifi_point_list, _p.reshape(-1,2)]) # update list
                        
                        wifi_rssi = wifi1.copy()
                        wifi_rssi.rssi += wifi_grad*i*grid_distance/d
                        
                        if tylor_approximation=='second':
                            wifi_rssi.rssi += wifi_grad*(i*grid_distance)**2/d**2
                        
                        if add_noises:
                            np.random.seed = random_state
                            wifi_rssi.rssi += np.random.normal(mu, sigma, [len(wifi_rssi.rssi)]) 

                        wifi_to_xy[tuple(_p)] = wifi_rssi
                    
    return wifi_to_xy, step_point_list, wifi_point_list


def make_wifi_feats(wp_list, wifi_to_xy):
    init_df = pd.DataFrame()
    for xy, _wifi in tqdm(wifi_to_xy.items()):
        df = pd.DataFrame(columns=['x','y'], index=[0])
        df.x, df.y = xy[0], xy[1]
        df[_wifi.bssid.values] = _wifi.rssi.values                
        init_df = init_df.append(df)
                
    return init_df


def get_train_wifi_feats(site, floor, wp_list, wifi_to_xy):
    try:
        train_wifi_feats = pd.read_csv(f'{extend_input_dir}/datas/{site}/{floor}/train_wifi_feats.csv')
    except:
        filenames = glob(f'{input_dir}/train/{site}/{floor}/*')
        train_wifi_feats = make_wifi_feats(wp_list, wifi_to_xy)
        
    return train_wifi_feats
    
    
def generate_wifi_points(
    train_wifi_feats,
    wp_list,
    add_wp_pair,
    max_range = 0.5,
    grid_distance = 0.1,
    add_noises = True,
    mu = 0,
    sigma = 0.1,
    random_state = 42
    ):
    
    _dict = {}
    for datas in train_wifi_feats.values:
        xy = datas[0:2]
        wifi = datas[2:].reshape(-1,len(datas)-2)
        for wp in wp_list:
            dist = np.sqrt((wp[0]-xy[0])**2 + (wp[1]-xy[1])**2)
            if dist<= max_range:
                wp = tuple(wp)
                if wp in _dict:
                    _dict[wp] = np.concatenate([_dict[wp], wifi])
                else:
                    _dict[wp] = wifi
                    
    init_df = pd.DataFrame()
    for wp_p in tqdm(add_wp_pair):
        wp1,wp2 = wp_p[0], wp_p[1]
        try:
            wifi1 = np.mean(_dict[wp1],axis=0)
            wifi2 = np.mean(_dict[wp2],axis=0)
        except:
            continue
        
        wifi_grad = wifi2 - wifi1
        d = np.sqrt((wp1[0]-wp2[0])**2 + (wp1[1]-wp2[1])**2)
        wp1 = np.array(wp1)
        wp2 = np.array(wp2)
        _datas = np.zeros((0,train_wifi_feats.shape[1]))
        for i in range(int(d//grid_distance)+1):
            _p = wp1 + (wp2-wp1)*i*grid_distance/d                    
            _wifi = wifi1 + wifi_grad*i*grid_distance/d
  
            if add_noises:
                np.random.seed = random_state
                _wifi += np.random.normal(mu, sigma, [len(wifi1)]) 
            
            __datas = np.concatenate([_p.reshape(-1,2), _wifi.reshape(-1,len(_wifi))], axis=1)
            _datas = np.concatenate([_datas, __datas])
        
        df = pd.DataFrame(_datas, columns= train_wifi_feats.columns)
        init_df = init_df.append(df)
        
    return init_df
       
       
def make_test_wifi_feats(
    test_path_list,
    train_wifi_feats
    ):
    
    init_df = pd.DataFrame()
    for i in test_path_list:
        path = path_pool[i]

        test_info, file_datas = helper_f.calibrate_test_path_timestamp(path, train_info) 
        wifi_datas = file_datas.wifi

        sep_tss = np.unique(wifi_datas[:, 0].astype(float))
        wifi_datas_list = compute_f.split_ts_seq(wifi_datas, sep_tss)
        for wifi_ds in wifi_datas_list:
            ts = float(wifi_ds[0,0])
    
            df = pd.DataFrame(columns = ['path','ts'], index=[0])
            df.path = i
            df.ts = ts
    
            wifi_df = pd.DataFrame(wifi_ds, columns=wifi_cols).astype(wifi_dtypes)
            wifi_df = wifi_df.groupby('bssid')['rssi'].mean().reset_index() 
            for bssid in wifi_df.bssid:
                df[bssid] = wifi_df[wifi_df.bssid == bssid].rssi.values[0]

            init_df = init_df.append(df)

    cols = ['path',"ts"] + list(train_wifi_feats.columns[2:])
    test = pd.DataFrame(columns = cols)
    test = test.append(init_df)[cols]
    
    return test