import sys

import torch
import numpy as np
import os
import pickle
import argparse
# for plotting
import matplotlib.pyplot as plt
import matplotlib


from torchvision import transforms

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from kalman_filter import KalmanFilterFT

import socket
from client import Client2
import time

import IPython
e = IPython.embed

from collect_data import RealSenseHandler 

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(camera_names, realsense_handler, serial_numbers, rand_crop_resize=False):

    frames_tuple = realsense_handler.get_latest_frames()  # Returns a tuple of 3 frames
    
    # Create a mapping from serial number to frame
    serial_to_frame = {}
    for idx, serial in enumerate(realsense_handler.serial_numbers):
        if idx < len(frames_tuple):
            serial_to_frame[serial] = frames_tuple[idx]
        else:
            serial_to_frame[serial] = None  # In case get_frames returns fewer frames

    curr_images = []

    for serial_number in serial_numbers:
        frame = serial_to_frame.get(serial_number)

        # Ensure frame is a NumPy array with shape (H, W, C)
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(2, 0, 1) 
        if isinstance(frame, np.ndarray):
            frame = np.transpose(frame, (2, 0, 1))
        elif not isinstance(frame, np.ndarray):
            print(f"[ERROR] Frame for camera {serial_number} is neither Tensor nor NumPy array. Skipping.")
            continue

        curr_images.append(frame)

    if not curr_images:
        print("No valid frames fetched for the specified camera_names. Returning None.")
        return None

    curr_image = np.stack(curr_images, axis=0)
    raw_image = torch.from_numpy(curr_image).float()
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0) # (1, num_cameras, C, H, W)
    

    if rand_crop_resize:
        # print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image, raw_image


def eval_bc(config, ckpt_name):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    ft = policy_config['ft']
    noleftcam = policy_config['noleftcam']
    #additional arg
    resp_te = policy_config['resp_te']
    filtering = policy_config['filtering']

    serial_numbers = ["f0210565", "f1380687", "f0244932"] # left, right, front
    realsense_handler = RealSenseHandler(serial_numbers, mode = 'inference')
    realsense_handler.start_frame_fetcher()

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if noleftcam:
        pre_process_force = lambda s_force: (s_force - stats['force_mean'][3:]) / stats['force_std'][3:]
        pre_process_torque = lambda s_torque: (s_torque - stats['torque_mean'][3:]) / stats['torque_std'][3:]
    else:
        pre_process_force = lambda s_force: (s_force - stats['force_mean']) / stats['force_std']
        pre_process_torque = lambda s_torque: (s_torque - stats['torque_mean']) / stats['torque_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # filtering
    if filtering:
        kf_force = KalmanFilterFT()
        kf_torque = KalmanFilterFT()
    

    query_frequency = policy_config['num_queries']
    # query_frequency = 15
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks


    ### evaluation 
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
        all_time_exp_weights = torch.zeros(max_timesteps).cuda()
        all_time_force = torch.zeros([max_timesteps, 3]).cuda()
        # store force norm for plotting
        force_norm = torch.zeros(max_timesteps).cuda()
        gripper_te_buffer = np.zeros(max_timesteps)
        gripper_curr_buffer = torch.zeros(max_timesteps).cuda()
        gripper_action_buffer = torch.zeros(max_timesteps).cuda()
    left, right, front = [], [], []
    qpos_set = []
    with torch.inference_mode():
        start_time = time.time()
        for t in range(max_timesteps):
            ### process previous timestep to get qpos and image_list
            data = client2.receive_data()
            while data is None:
                print(f"[INFO] Waiting for qpos, force, torque at timestep {t}...")
                data = client2.receive_data()
            if ft:
                print("using ft data...")
                qpos_numpy = data[:16]
                force_numpy = data[16:22]
                torque_numpy = data[22:]
                # filtering
                if filtering:
                    force = kf_force.update(force_numpy)
                    torque = kf_torque.update(torque_numpy)
                force = pre_process_force(force)
                torque = pre_process_torque(torque)
                # force = pre_process_force(force_numpy)
                # torque = pre_process_torque(torque_numpy)
                # # filtering
                # if filtering:
                #     force = kf_force.update(force)
                #     torque = kf_torque.update(torque)
                force = torch.from_numpy(force).float().cuda().unsqueeze(0)
                torque = torch.from_numpy(torque).float().cuda().unsqueeze(0)
            else:
                qpos_numpy = data[:16]
                force = None
                torque = None
            
            if resp_te:
                all_time_force[t,:] = force.squeeze(0)[:3]

            qpos = pre_process_qpos(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            # print(f"force : {force}, torque : {torque}")
            
            if t % query_frequency == 0:
                curr_image, raw_image = get_image(camera_names, realsense_handler, serial_numbers, rand_crop_resize=(config['policy_class'] == "Diffusion"))
                l,r,f = raw_image[0], raw_image[1], raw_image[2]
                left.append(l)
                right.append(r)
                front.append(f)
            # curr_image = get_image(ts, camera_names)
            if t == 0:
                # warm up
                for _ in range(10):
                    policy(qpos, force, torque, curr_image)
                print('network warm up done')

            start_t = time.time()
            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, force, torque, curr_image)
                    gripper_action_buffer[t] = all_actions[0][0][7]
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t] # This gathers the actions that were predicted at earlier steps for time t.
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    # if resp_te:
                    #     alpha = 10
                    #     start_idx = max(0, t + 1 - len(actions_for_curr_step))
                    #     force_avg = all_time_force[start_idx:t+1,:].mean(dim=0)
                    #     all_time_exp_weights[t] = torch.norm(force-force_avg)
                    #     exp_weights = []
                    #     for i in range(len(actions_for_curr_step)):
                    #         exp_weights[i] = np.exp(-k * i + alpha*all_time_exp_weights[t-(len(actions_for_curr_step)-i-1)])

                    if resp_te:
                        alpha = 5 # success alpha 10
                        start_idx = max(0, t + 1 - len(actions_for_curr_step))
                        force_avg = all_time_force[start_idx:t+1,:].mean(dim=0)
                        all_time_exp_weights[t] = torch.norm(force.squeeze(0)[:3]-force_avg)
                        # print(f"force diff : {force.squeeze(0)[:3]-force_avg}")
                        print(f"force diff norm : {torch.norm(force.squeeze(0)[:3]-force_avg)}")
                        # store data for plotting
                        force_norm[t] = torch.norm(force.squeeze(0)[:3]-force_avg)
                        # Initialize exp_weights with the correct size
                        exp_weights = torch.zeros(len(actions_for_curr_step), device=force.device)
                        
                        for i in range(len(actions_for_curr_step)):
                            # Stay in PyTorch for all calculations
                            weight_idx = t-(len(actions_for_curr_step)-i-1)
                            exp_weights[i] = torch.exp(-k * i + alpha * all_time_exp_weights[weight_idx])
                    else:
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = exp_weights.cuda().unsqueeze(dim=1) if resp_te else torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    # print(f"(in inference.py) exp_weights: {exp_weights}")
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    gripper_curr_buffer[t] = actions_for_curr_step[-1][7]
                else:
                    raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "Diffusion":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, curr_image)
            else:
                raise NotImplementedError
            
            end_t = time.time()
            print(f"policy computation time: {(end_t - start_t):.2f}")
            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()  # (1, 14) -> (14,)
            action = post_process(raw_action)  # normalization
            target_qpos = action
            # target_qpos[7:] = 0
            gripper_te_buffer[t] = target_qpos[7]
            qpos_set.append(target_qpos)
            
            success = client2.send_action(target_qpos)

            if not success:
                print(f"[WARNING] Failed to send action at timestep {t}.")
        plt.plot(force_norm.cpu().numpy())
        plt.plot(gripper_te_buffer)
        plt.plot(gripper_curr_buffer.cpu().numpy())
        plt.plot(gripper_action_buffer.cpu().numpy())
        plt.xlabel('Time Step')
        plt.ylabel('Force norm')
        plt.show()
        qpos_set = np.array(qpos_set, dtype=np.float32)


            

def main(args_list=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, default='GACT', help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--eval_every', action='store', type=int, help='eval_every', required=True)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--ft', action='store', type=int, default=0, help='ft sensor')
    parser.add_argument('--noleftcam', action='store', type=int, default=0, help='Not using leftcam image')
    # additional arg
    parser.add_argument('--resp_te', action='store', type=int, default=0, help='responsive temporal ensemble')
    parser.add_argument('--filtering', action='store', type=int, default=0, help='filtering')
    parser.add_argument('--ft_softmax', action='store', type=int, default=0, help='softmax force&torque')


    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, default=100, help='chunk_size', required=False)
    
    # Additional arguments
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help="Backbone learning rate")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="Weight decay")
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')

    # Model parameters
    parser.add_argument('--backbone', default='resnet18', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, replaces stride with dilation in last conv block")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding")
    parser.add_argument('--camera_names', default=['left', 'right', 'front'], type=list, help="A list of camera names")
    
    # * Graph Transformer
    parser.add_argument('--env_name', type=str, default='rby1', help='Name of the environment')
    parser.add_argument('--state_dim', type=int, default=17, help='Dimensionality of the joint state')
    parser.add_argument('--enc_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward network dimension in transformer')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in transformer')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads in transformer')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--shared_film', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    if args_list is None:
        args = vars(parser.parse_args())
    else:
        args = vars(parser.parse_args(args_list))

    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    num_epochs = args['num_epochs']

    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        # from real_constants import TASK_CONFIGS
        # task_config = TASK_CONFIGS[task_name]
        pass
    else:
        # from aloha_scripts.constants import TASK_CONFIGS
        print('real robot tasks execution')
        from constants_rby1 import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    if policy_class == 'ACT':
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': args['lr_backbone'],
                         'backbone': args['backbone'],
                         'enc_layers': args['enc_layers'],
                         'dec_layers': args['dec_layers'],
                         'nheads': args['nheads'],
                         'camera_names': camera_names,
                         'ft': args['ft'],
                         'noleftcam': args['noleftcam'],
                         # additional arg
                         'resp_te': args['resp_te'],
                         'filtering': args['filtering']
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': args['lr_backbone'], 'backbone' : args['backbone'], 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': args['state_dim'],
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    # ckpt_name = "policy_epoch_10000_seed_0.ckpt"
    ckpt_name = "policy_epoch_30000_seed_0.ckpt"
    # ckpt_name = "policy_last.ckpt"

    eval_bc(config, ckpt_name)

if __name__ == "__main__":
    SERVER_HOST = "192.168.200.132" 
    SERVER_PORT = 8001                

    client2 = Client2(SERVER_HOST, SERVER_PORT)

    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Close sockets gracefully
        client2.close()
