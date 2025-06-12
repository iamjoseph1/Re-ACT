# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_ft_data(hdf5_file):
#     with h5py.File(hdf5_file, 'r') as f:
#         # Extract force and torque data
#         force_data = f['/observation/force'][:]
#         torque_data = f['/observation/torque'][:]
        
#         # Ensure data is structured as (N, 6) for 6D force/torque readings
#         if force_data.shape[1] != 6 or torque_data.shape[1] != 6:
#             raise ValueError("Unexpected data shape, expected (N, 6)")
        
#         # Use index as time step
#         time_steps = np.arange(force_data.shape[0])
        
#         # Separate left and right forces/torques
#         force_left = force_data[:, :3]
#         force_right = force_data[:, 3:]
#         torque_left = torque_data[:, :3]
#         torque_right = torque_data[:, 3:]
        
#         # Plot left and right forces
#         plt.figure(figsize=(12, 6))
#         labels_force = ['Fx', 'Fy', 'Fz']
#         labels_torque = ['Tx', 'Ty', 'Tz']
        
#         plt.subplot(2, 1, 1)
#         for i in range(3):
#             plt.plot(time_steps, force_left[:, i], label=f"Left {labels_force[i]}")
#             plt.plot(time_steps, force_right[:, i], '--', label=f"Right {labels_force[i]}")
#         plt.xlabel('Time Step')
#         plt.ylabel('Force')
#         plt.title('Left vs Right Force Data')
#         plt.legend()
#         plt.grid()
        
#         # Plot left and right torques
#         plt.subplot(2, 1, 2)
#         for i in range(3):
#             plt.plot(time_steps, torque_left[:, i], label=f"Left {labels_torque[i]}")
#             plt.plot(time_steps, torque_right[:, i], '--', label=f"Right {labels_torque[i]}")
#         plt.xlabel('Time Step')
#         plt.ylabel('Torque')
#         plt.title('Left vs Right Torque Data')
#         plt.legend()
#         plt.grid()
        
#         plt.tight_layout()
#         plt.show()


# demo_dir = "/media/rby1/T7/dataset/rby1_click_pen_ft/episode_1.hdf5" # Replace with your actual file
# plot_ft_data(demo_dir)

# Example usage
# hdf5_filename = 'your_data.hdf5'  # Replace with actual file
# plot_ft_data(hdf5_filename)



import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_ft_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        # Extract force data
        ft_data = f['/observations/torque'][:]  # Modify if dataset name differs
        
        # # Ensure ft_data is structured as (N, 6) for 6D force readings
        # if ft_data.shape[1] != 6:
        #     raise ValueError("Unexpected f/t data shape, expected (N, 6)")
        
        # # Extract torque data
        # ft_data = f['/observations/torque'][0:3]  # Modify if dataset name differs
        
        # # Ensure ft_data is structured as (N, 6) for 6D torque readings
        # if ft_data.shape[1] != 6:
        #     raise ValueError("Unexpected f/t data shape, expected (N, 6)")
        
        # Use index as time step
        time_steps = np.arange(ft_data.shape[0])
        
        # Plot force and torque separately
        labels = ['Fx_left', 'Fy_left', 'Fz_left', 'Tx_left', 'Ty_left', 'Tz_left']
        plt.figure(figsize=(10, 6))
        
        for i in range(6):
            plt.plot(time_steps, ft_data[:, i], label=labels[i])
        
        plt.xlabel('Time Step')
        plt.ylabel('Force/Torque')
        plt.title('Force/Torque Data Over Time Steps')
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
demo_dir = "/media/rby1/T7/dataset/rby1_click_pen_ft/episode_1.hdf5" # Replace with your actual file
plot_ft_data(demo_dir)

