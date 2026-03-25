import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_future_traj, generate_future_ego
from tqdm import tqdm
import copy

system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds."

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
parser.add_argument("--future", action='store_true')
parser.add_argument("--llm_kd", action='store_true')
parser.add_argument("--one_point", action='store_true')
parser.add_argument("--delta", action='store_true')
args = parser.parse_args()

data = pickle.load(open('./data/nuscenes/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('./data/nuscenes/full_split.json', 'r'))
tokens = split[args.split]

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

dataroot = '/workspace/group_share/adc-perception-mlinfra/caojj3/data/nuscenes'
train_messages = []

for token in tqdm(tokens):
    future_traj = generate_future_traj(data, token, args)
    user_message, images_path = generate_user_message(args, data, token)                      
    if args.llm_kd:
        user_message_student, images_path_student = generate_user_message(args, data, token)
        teacher_args = copy.deepcopy(args)
        teacher_args.future = True
        user_message_teacher, images_path_teacher = generate_user_message(teacher_args, data, token)
        future_ego = generate_future_ego(data, token, args.split)
        train_message = {
                            "id": token,
                            "images": images_path_student,
                            "system": system,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n" + user_message_student + "Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds.\n",                              
                                },
                                {
                                    "role": "assistant",                                
                                    "content": future_traj + " These are the future waypoints. \n" 
                                },                    
                            ],
                            "teacher_images": images_path_teacher,
                            "teacher_system": system,
                            "teacher_messages": [
                                {
                                    "role": "user",
                                    "content": "Here are current front cam image from the car: 'CURRENT CAM_FRONT': <image>\n Here are future three secends front cam image from the car: 'FUTURE 1S CAM_FRONT': <image>\n, 'FUTURE 2S CAM_FRONT': <image>\n, 'FUTURE 3S CAM_FRONT': <image>\n" + user_message_teacher + future_ego + "Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds.\n"                                
                                },
                                {
                                    "role": "assistant",                                
                                    "content": future_traj + " These are the future waypoints. \n"
                                },                    
                            ]
                        }
    elif args.future:
        future_ego = generate_future_ego(data, token, args.split)
        system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds."
        train_message = {
                            "id": token,
                            "images": images_path,
                            "system": system,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Here are current front cam image from the car: 'CURRENT CAM_FRONT': <image>\n Here are future three secends front cam image from the car: 'FUTURE 1S CAM_FRONT': <image>\n" "'FUTURE 2S CAM_FRONT': <image>\n" "'FUTURE 3S CAM_FRONT': <image>\n" + user_message + "Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds.\n"                                
                                },
                                {
                                    "role": "assistant",                                
                                    "content": future_traj + " These are the future waypoints. \n"
                                },                    
                            ]
                        } 
    else:
        train_message = {
                            "id": token,
                            "images": images_path,
                            "system": system,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Here are current six images from the car: 'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n, 'CAM_FRONT_RIGHT': <image>\n,'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n, 'CAM_BACK_RIGHT': <image>\n" + user_message + "Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds.\n",                              
                                },
                                {
                                    "role": "assistant",                                
                                    "content": future_traj + " These are the future movement difference waypoints. \n" 
                                },                    
                            ]                            
                        }
    train_messages.append(train_message)
if args.future:
    with open(f"./data/nuscenes/Drive_KD_{args.split}_his_ego_future.json", "w") as f:
        json.dump(train_messages, f, indent=4)  
elif args.llm_kd:
    with open(f"./data/nuscenes/Drive_KD_{args.split}_his_ego_llm_kd.json", "w") as f:
        json.dump(train_messages, f, indent=4)  
else:
    with open(f"./data/nuscenes/Drive_KD_{args.split}_his_ego.json", "w") as f:
        json.dump(train_messages, f, indent=4)