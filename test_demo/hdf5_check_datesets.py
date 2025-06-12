import h5py

def list_datasets(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        print("Datasets in the file:")
        def print_name(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name)
        f.visititems(print_name)

# Example usage
demo_dir = "/media/rby1/T7/dataset/rby1_click_pen_ft/episode_1.hdf5" # Replace with your actual file
list_datasets(demo_dir)
