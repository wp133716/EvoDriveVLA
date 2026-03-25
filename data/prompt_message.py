import numpy as np
from mmengine.fileio import load
import os
import bisect

def get_steer(timestamp, scene_name):
    can_bus_path = f'./data/nuscenes/can_bus/{scene_name}_steeranglefeedback.json'
    if os.path.exists(can_bus_path):
        can_bus = load(can_bus_path, file_format='json')
        can_bus = list(sorted(can_bus, key=lambda x: x['utime']))
        times = [entry['utime'] for entry in can_bus]
        idx = bisect.bisect_left(times, timestamp)
        if idx == len(times):
            idx-=1
        return round(can_bus[idx]['value'], 2)
    else:
        return 0.0

def get_can_bus(sample):
    can_bus = sample['can_bus']
    can_bus = np.round(can_bus, 2)
    data = {}
    data['acc_x'] = can_bus[7]
    if can_bus[8] != 0.0:
        data['acc_y'] = -can_bus[8]
    else:
        data['acc_y'] = can_bus[8]
    data['vel'] = can_bus[13]
    if 'scene_token' in sample:
        scene_file = load('./data/nuscenes/v1.0-trainval/scene.json', file_format='json')
        scene_name = next(item['name'] for item in scene_file if item['token'] == sample['scene_token'])
        data['steer'] = get_steer(sample['timestamp'], scene_name)
        if data['steer'] == -0.0:
            data['steer'] = 0.0
    else: 
        data['steer'] = 0.0
    return data

def generate_user_message(args, data, token, perception_range=20.0, short=True):

    # user_message  = f"You have received new input data to help you plan your route.\n"
    user_message  = f"Here's some information you'll need:\n"
    
    data_dict = data[token]
    if args.future or args.one_point:
        camera_types = [
            'CAM_FRONT',
        ]
    else:
        camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]  
    images_path = []
    for cam in camera_types:
        images_path.append(data_dict['cams'][cam]['data_path'].replace('/localdata_ssd/nuScenes', 'nuscenes', 1))
    if args.future:
        future_cams = []
        sample = data_dict
        for i in range(6):
            if sample['next'] == '':
                break
            next_sample = data[sample['next']]
            if i % 2 == 1:
                future_cams.append(next_sample['cams']['CAM_FRONT']['data_path'].replace('/localdata_ssd/nuScenes', 'nuscenes', 1))  
            sample = next_sample
        
        if len(future_cams) < 3:
            for _ in range(1 - len(future_cams)):
                future_cams.append(sample['cams']['CAM_FRONT']['data_path'].replace('/localdata_ssd/nuScenes', 'nuscenes', 1))
        images_path.extend(future_cams)
    if args.one_point:
        future_cams = []
        sample = data_dict
        if sample['next'] == '' or sample['next'] is None:
            next_sample = sample
        else:
            next_sample = data[sample['next']]
        future_cams.append(next_sample['cams']['CAM_FRONT']['data_path'].replace('/localdata_ssd/nuScenes', 'nuscenes', 1))
        images_path.extend(future_cams)
    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    xh5 = data_dict['gt_ego_his_trajs'][4][0]
    yh5 = data_dict['gt_ego_his_trajs'][4][1]
    user_message += f"Historical Trajectory (last 2 seconds):"
    user_message += f" [(-2.0s):({xh1:.2f},{yh1:.2f}), (-1.5s):({xh2:.2f},{yh2:.2f}), (-1.0s):({xh3:.2f},{yh3:.2f}), (-0.5s):({xh4:.2f},{yh4:.2f}), (-0.0s):({xh5:.2f},{yh5:.2f})]\n"
    """
    Historical Ego
    """
    pre = 0
    the_sample = data_dict
    his_ego = ''
    ego_list = []
    can_bus = None
    while pre<4 and the_sample['prev'] != '':
        pre+=1
        pre_id = the_sample['prev']
        the_sample = data[pre_id]
        can_bus = get_can_bus(the_sample)
        ego_list.append(can_bus)
    user_message += f"Historical ego (last 2 seconds), Format: (Velocity, Acceleration_x, Acceleration_y, steer): "
    if len(ego_list) <= 2:
        vel = ((xh4 - xh1)**2 + (yh4 - yh1)**2)**0.5 / 1.5
        acc_x = (xh4 - xh3 - xh2 + xh1) / 0.5
        acc_y = (yh4 - yh3 - yh2 + yh1) / 0.5
        steer = 0.0
        can_bus = {'vel': vel, 'acc_x': acc_x, 'acc_y': acc_y, 'steer': steer}
    current_ego = get_can_bus(data_dict)
    if len(ego_list) < 4:
        for _ in range(4 - len(ego_list)):
            ego_list.append(can_bus)
    user_message += f"[(-2.0s):({ego_list[3]['vel']:.2f} m/s, {ego_list[3]['acc_x']:.2f} m/s^2, {ego_list[3]['acc_y']:.2f} m/s^2, {ego_list[3]['steer']:.2f}), (-1.5s):({ego_list[2]['vel']:.2f} m/s, {ego_list[2]['acc_x']:.2f} m/s^2, {ego_list[2]['acc_y']:.2f} m/s^2, {ego_list[2]['steer']:.2f}), (-1.0s):({ego_list[1]['vel']:.2f} m/s, {ego_list[1]['acc_x']:.2f} m/s^2, {ego_list[1]['acc_y']:.2f} m/s^2, {ego_list[1]['steer']:.2f}), (-0.5s):({ego_list[0]['vel']:.2f} m/s, {ego_list[0]['acc_x']:.2f} m/s^2, {ego_list[0]['acc_y']:.2f} m/s^2, {ego_list[0]['steer']:.2f}), (-0.0s):({current_ego['vel']:.2f} m/s, {current_ego['acc_x']:.2f} m/s^2, {current_ego['acc_y']:.2f} m/s^2, {current_ego['steer']:.2f})]\n"
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "RIGHT"
    elif left > 0:
        mission_goal = "LEFT"
    else:
        assert forward > 0
        mission_goal = "FORWARD"
    user_message += f"Mission Goal: "
    user_message += f"{mission_goal}\n"

    """
    Traffic Rule
    """
    user_message += f"Traffic Rules: Avoid collision with other objects.\n- Always drive on drivable regions.\n- Avoid driving on occupied regions.\n"
    
    return user_message, images_path


def generate_future_traj(data, token, args):
    data_dict = data[token]
    assitant_message = ""

    x1 = data_dict['gt_ego_fut_trajs'][1][0]
    x2 = data_dict['gt_ego_fut_trajs'][2][0]
    x3 = data_dict['gt_ego_fut_trajs'][3][0]
    x4 = data_dict['gt_ego_fut_trajs'][4][0]
    x5 = data_dict['gt_ego_fut_trajs'][5][0]
    x6 = data_dict['gt_ego_fut_trajs'][6][0]
    y1 = data_dict['gt_ego_fut_trajs'][1][1]
    y2 = data_dict['gt_ego_fut_trajs'][2][1]
    y3 = data_dict['gt_ego_fut_trajs'][3][1]
    y4 = data_dict['gt_ego_fut_trajs'][4][1]
    y5 = data_dict['gt_ego_fut_trajs'][5][1]
    y6 = data_dict['gt_ego_fut_trajs'][6][1]
    if args.one_point:
        assitant_message += f"[({x1:.2f},{y1:.2f}), (0.00,0.00), (0.00,0.00), (0.00,0.00), (0.00,0.00), (0.00,0.00)]"
        return assitant_message
    elif args.delta:
        assitant_message += f"[({x1:.2f},{y1:.2f}), ({(x2-x1):.2f},{(y2-y1):.2f}), ({(x3 - x2):.2f},{(y3 - y2):.2f}), ({(x4 - x3):.2f},{(y4 - y3):.2f}), ({(x5 - x4):.2f},{(y5 - y4):.2f}), ({(x6 - x5):.2f},{(y6 - y5):.2f})]"
        return assitant_message
    assitant_message += f"[({x1:.2f},{y1:.2f}), ({x2:.2f},{y2:.2f}), ({x3:.2f},{y3:.2f}), ({x4:.2f},{y4:.2f}), ({x5:.2f},{y5:.2f}), ({x6:.2f},{y6:.2f})]"
    return assitant_message

def generate_future_ego(data, token, type):
    the_data = data[token]
    if the_data['next'] == None or the_data['next'] == '':
        next1 = the_data
    else:   
        next1 = data[the_data['next']]
    if next1['next'] == None or next1['next'] == '':
        next2 = next1
    else:       
        next2 = data[next1['next']]
    if next2['next'] == None or next2['next'] == '':
        next3 = next2
    else:
        next3 = data[next2['next']]
    if next3['next'] == None or next3['next'] == '':
        next4 = next3
    else:
        next4 = data[next3['next']]
    if next4['next'] == None or next4['next'] == '':
        next5 = next4
    else:
        next5 = data[next4['next']]
    if next5['next'] == None or next5['next'] == '':
        next6 = next5
    else:
        next6 = data[next5['next']]
    ego1 = get_can_bus(next1)
    ego2 = get_can_bus(next2)
    ego3 = get_can_bus(next3)
    ego4 = get_can_bus(next4)
    ego5 = get_can_bus(next5)
    ego6 = get_can_bus(next6)
    assitant_message = "Future ego (next 3 seconds), Format: (Velocity, Acceleration_x, Acceleration_y, steer): "
    assitant_message += f"[(0.5s):({ego1['vel']:.2f} m/s, {ego1['acc_x']:.2f} m/s^2, {ego1['acc_y']:.2f} m/s^2, {ego1['steer']:.2f}), "
    assitant_message += f"(1.0s):({ego2['vel']:.2f} m/s, {ego2['acc_x']:.2f} m/s^2, {ego2['acc_y']:.2f} m/s^2, {ego2['steer']:.2f}), "
    assitant_message += f"(1.5s):({ego3['vel']:.2f} m/s, {ego3['acc_x']:.2f} m/s^2, {ego3['acc_y']:.2f} m/s^2, {ego3['steer']:.2f}), "
    assitant_message += f"(2.0s):({ego4['vel']:.2f} m/s, {ego4['acc_x']:.2f} m/s^2, {ego4['acc_y']:.2f} m/s^2, {ego4['steer']:.2f}), "
    assitant_message += f"(2.5s):({ego5['vel']:.2f} m/s, {ego5['acc_x']:.2f} m/s^2, {ego5['acc_y']:.2f} m/s^2, {ego5['steer']:.2f}), "
    assitant_message += f"(3.0s):({ego6['vel']:.2f} m/s, {ego6['acc_x']:.2f} m/s^2, {ego6['acc_y']:.2f} m/s^2, {ego6['steer']:.2f})]\n"
    return assitant_message



