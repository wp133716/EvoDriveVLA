import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import os
import re
import ast
import argparse

cache_data = pickle.load(open('./data/nuscenes/cached_nuscenes_info.pkl', 'rb'))

def planning_evaluation_uniad(pred_trajs_dict, config):
    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    from metric_uniad import PlanningMetric
    with open(Path(os.path.join(config.gt_folder, 'uniad_gt_seg.pkl')),'rb') as f:
        gt_occ_map = pickle.load(f)
    for token in gt_occ_map.keys():
        if not isinstance(gt_occ_map[token], torch.Tensor):
            gt_occ_map[token] = torch.tensor(gt_occ_map[token])
    
    metric_planning_val = PlanningMetric(ts).to(device)     

    with open(Path(os.path.join(config.gt_folder, 'gt_traj.pkl')),'rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open(Path(os.path.join(config.gt_folder, 'gt_traj_mask.pkl')),'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    for index, token in enumerate(tqdm(gt_trajs_dict.keys())):

        gt_trajectory =  torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)

        try:
            data = re.findall(r'\((\-?\d+\.\d+),(\-?\d+\.\d+)\)', pred_trajs_dict[token])
            result = [(float(x), float(y)) for x, y in data]          
            if len(data) == 0:
                data = re.findall(r'[\(\[]\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*[\)\]]', pred_trajs_dict[token])
                result = [(-float(y), float(x)) for x, y in data]
            output_trajs = torch.tensor(result)
            output_trajs = output_trajs.reshape(gt_traj_mask.shape)
            output_trajs = output_trajs.to(device)
        except:
            continue
        

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        if output_trajs.shape[1] % 2: # in case the current timestep is inculded
            output_trajs = output_trajs[:, 1:]

        if occupancy.shape[1] % 2: # in case the current timestep is inculded
            occupancy = occupancy[:, 1:]
        
        if gt_trajectory.shape[1] % 2: # in case the current timestep is inculded
            gt_trajectory = gt_trajectory[:, 1:]

        if gt_traj_mask.shape[1] % 2:  # in case the current timestep is inculded
            gt_traj_mask = gt_traj_mask[:, 1:]
        
        metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], token, gt_traj_mask)
          
    results = {}
    scores = metric_planning_val.compute()
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value[:(i+1)*2].mean()

    method = (config.method, 
            "{:.4f}".format(scores["L2"][1]), \
            "{:.4f}".format(scores["L2"][3]), \
            "{:.4f}".format(scores["L2"][5]),\
            "{:.4f}".format((scores["L2"][1]+ scores["L2"][3]+ scores["L2"][5]) / 3.), \
            "{:.4f}".format(scores["obj_box_col"][1]*100), \
            "{:.4f}".format(scores["obj_box_col"][3]*100), \
            "{:.4f}".format(scores["obj_box_col"][5]*100), \
            "{:.4f}".format(100*(scores["obj_box_col"][1]+ scores["obj_box_col"][3]+ scores["obj_box_col"][5]) / 3.), \
            )
    return method

def planning_evaluation_stp3(pred_trajs_dict, config):
    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    from metric_stp3 import PlanningMetric
    with open(Path(os.path.join(config.gt_folder, 'stp3_gt_seg.pkl')),'rb') as f:
        gt_occ_map = pickle.load(f)
    for token in gt_occ_map.keys():
        if not isinstance(gt_occ_map[token], torch.Tensor):
            gt_occ_map[token] = torch.tensor(gt_occ_map[token])
        gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
        gt_occ_map[token] = torch.flip(gt_occ_map[token], [-2])
    
    metric_planning_val = PlanningMetric(ts).to(device)     

    with open(Path(os.path.join(config.gt_folder, 'gt_traj.pkl')),'rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open(Path(os.path.join(config.gt_folder, 'gt_traj_mask.pkl')),'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    for index, token in enumerate(tqdm(gt_trajs_dict.keys())):

        gt_trajectory =  torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)

        try:
            data = re.findall(r'\((\-?\d+\.\d+),(\-?\d+\.\d+)\)', pred_trajs_dict[token])
            result = [(float(x), float(y)) for x, y in data]
            if len(data) == 0:
                data = re.findall(r'[\(\[]\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*[\)\]]', pred_trajs_dict[token])
                result = [(-float(y), float(x)) for x, y in data]
            output_trajs = torch.tensor(result)
            output_trajs = output_trajs.reshape(gt_traj_mask.shape)
            output_trajs = output_trajs.to(device)
        except:
            continue
        

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        if output_trajs.shape[1] % 2: # in case the current timestep is inculded
            output_trajs = output_trajs[:, 1:]

        if occupancy.shape[1] % 2: # in case the current timestep is inculded
            occupancy = occupancy[:, 1:]
        
        if gt_trajectory.shape[1] % 2: # in case the current timestep is inculded
            gt_trajectory = gt_trajectory[:, 1:]

        if gt_traj_mask.shape[1] % 2:  # in case the current timestep is inculded
            gt_traj_mask = gt_traj_mask[:, 1:]
        
        metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], token, gt_traj_mask)
          
    results = {}
    scores = metric_planning_val.compute()
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value[:(i+1)*2].mean()
    
    method = (config.method,    "{:.4f}".format(results["plan_L2_1s"]), \
                                "{:.4f}".format(results["plan_L2_2s"]), \
                                "{:.4f}".format(results["plan_L2_3s"]), \
                                "{:.4f}".format((results["plan_L2_1s"]+results["plan_L2_2s"]+results["plan_L2_3s"])/3.), \
                                "{:.4f}".format(results["plan_obj_box_col_1s"]*100), \
                                "{:.4f}".format(results["plan_obj_box_col_2s"]*100), \
                                "{:.2f}".format(results["plan_obj_box_col_3s"]*100), \
                                "{:.4f}".format(((results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results["plan_obj_box_col_3s"])/3)*100), \
                                )
    return method   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation of planning')
    parser.add_argument('--method', type=str, help='name of the method being evaluated, used for table print', default='Drive_KD')
    parser.add_argument('--result_file', type=str, help='path to the result file')
    parser.add_argument('--save_file', type=str, help='path to the eval_result file')
    parser.add_argument('--gt_folder', type=str, default='./data/nuscenes/metrics')
    config = parser.parse_args()

    result_file = Path(config.result_file)
    pred_trajs_list = json.load(open(config.result_file, 'r'))
    pred_trajs_dict = {d['id']: {k: v for k, v in d.items() if k != 'id'}['predict'] for d in pred_trajs_list}
    stp3 = planning_evaluation_stp3(pred_trajs_dict, config)
    uniad = planning_evaluation_uniad(pred_trajs_dict, config)
    metric_result = {}
    metric_result['stp3'] = {'L2(m)': {'1s': stp3[1], '2s': stp3[2], '3s': stp3[3], 'Avg.': stp3[4]},
                            'Collision(%)': {'1s': stp3[5], '2s': stp3[6], '3s': stp3[7], 'Avg.': stp3[8]},
                            }
    metric_result['uniad'] = {'L2(m)': {'1s': uniad[1], '2s': uniad[2], '3s': uniad[3], 'Avg.': uniad[4]},
                            'Collision(%)': {'1s': uniad[5], '2s': uniad[6], '3s': uniad[7], 'Avg.': uniad[8]},
                            }
    print(json.dumps(metric_result, indent=4))
    with open(config.save_file, 'w') as f:
        json.dump(metric_result, f, indent=4)
