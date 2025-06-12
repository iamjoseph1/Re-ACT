import h5py
import matplotlib.pyplot as plt
import os

def load_ft_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        # Extract force/torque data
        f_data = f['/observations/force'][:]  # Modify if dataset name differs
        t_data = f['/observations/torque'][:]  # Modify if dataset name differs
        

    return f_data, t_data

hdf5_filename_right = "/media/rby1/T7/dataset/rby1_box_wiping_ft/episode_2.hdf5"  # Replace with actual file

f_measurements, t_measurements=load_ft_data(hdf5_filename_right)

hdf5_filename_left = "/media/rby1/T7/dataset/rby1_box_wiping_ft/episode_2.hdf5"  # Replace with actual file

f_measurements_org, t_measurements_org = load_ft_data(hdf5_filename_left)

# vis_dir = '/media/rby1/T7/dataset/rby1_heavylight_ft/visual'

fig, axes = plt.subplots(2, 6, figsize=(16, 6))  # 2 rows, 6 columns

# Labels for force and torque measurements
force_labels = ["Fx_left", "Fy_left", "Fz_left", "Fx_right", "Fy_right", "Fz_right"]
torque_labels = ["Mx_left", "My_left", "Mz_left", "Mx_right", "My_right", "Mz_right"]

# Plot force measurements (Row 0)
for i, label in enumerate(force_labels):
    ax = axes[0, i]  # First row
    ax.plot(f_measurements_org[:, i], label="Light Data", linewidth=2)
    ax.plot(f_measurements[:, i], label="heavy Data", alpha=1)
    ax.set_title(label)
    ax.legend()

# Plot torque measurements (Row 1)
for i, label in enumerate(torque_labels):
    ax = axes[1, i]  # Second row
    ax.plot(t_measurements_org[:, i], label="Light Data", linewidth=2)
    ax.plot(t_measurements[:, i], label="heavy Data", alpha=1)
    ax.set_title(label)
    ax.legend()

# save plot
# plt.savefig(os.path.join(vis_dir, 'comparison.png'))
plt.tight_layout()
plt.show()