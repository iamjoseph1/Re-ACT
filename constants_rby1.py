import pathlib

### Task parameters
# DATA_DIR = '<put your data dir here>'
# DATA_DIR = '/home/rby1/G-ACT/G-ACT/dataset'
DATA_DIR = '/media/rby1/T7/dataset'
TASK_CONFIGS = {
    'rby1_tissue':{
        'dataset_dir': DATA_DIR + '/rby1_tissue_hdf5',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left', 'right', 'front']
    },
    'rby1_transfer':{
        'dataset_dir': DATA_DIR + '/rby1_transfer_l515_line_cropped',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left', 'right', 'front']
    },
    'rby1_click_pen_cropped':{
        'dataset_dir': DATA_DIR + '/rby1_click_pen_cropped',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['left', 'right', 'front']
    },
    'rby1_click_pen_ft':{
    'dataset_dir': DATA_DIR + '/rby1_click_pen_ft_3',
    'num_episodes': 50,
    'episode_len': 400,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_heavylight_ft':{
    'dataset_dir': DATA_DIR + '/rby1_heavylight_ft',
    'num_episodes': 100,
    'episode_len': 400,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_box_pulling_ft':{
    'dataset_dir': DATA_DIR + '/rby1_box_pulling_ft',
    'num_episodes': 100,
    'episode_len': 400,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_box_pulling_right_ft':{
    'dataset_dir': DATA_DIR + '/rby1_box_pulling_right_ft',
    'num_episodes': 100,
    'episode_len': 400,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_box_pulling_aftergrip_ft':{
    'dataset_dir': DATA_DIR + '/rby1_box_pulling_aftergrip_ft',
    'num_episodes': 60,
    'episode_len': 200,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_box_pulling_aftergrip2_ft':{
    'dataset_dir': DATA_DIR + '/rby1_box_pulling_aftergrip2_ft',
    'num_episodes': 100,
    'episode_len': 200,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_box_pulling_aftergrip3_ft':{
    'dataset_dir': DATA_DIR + '/rby1_box_pulling_aftergrip3_ft',
    'num_episodes': 100,
    'episode_len': 200,
    'camera_names': ['left', 'right', 'front']
    },

    'rby1_box_pulling_withgrip_ft':{
    'dataset_dir': DATA_DIR + '/rby1_box_pulling_withgrip_ft',
    'num_episodes': 100,
    'episode_len': 400, #300
    'camera_names': ['left', 'right', 'front']
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
