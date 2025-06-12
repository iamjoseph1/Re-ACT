import os
import re
import glob

def rename_episode_files(directory='.'):
    """
    Renames episode HDF5 files to have consecutive indices starting from 0.
    
    Args:
        directory: The directory containing the HDF5 files (default: current directory)
    """
    # Find all episode files with the pattern "episode_X.hdf5"
    file_pattern = os.path.join(directory, "episode_*.hdf5")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files matching pattern 'episode_*.hdf5' found in {directory}")
        return
    
    # Extract indices and create a mapping of (index, filename)
    file_indices = []
    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.match(r'episode_(\d+)\.hdf5', filename)
        if match:
            index = int(match.group(1))
            file_indices.append((index, file_path))
    
    # Sort by the original index
    file_indices.sort()
    
    print(f"Found {len(file_indices)} episode files to rename")
    
    # Create a temporary directory to avoid naming conflicts during renaming
    temp_dir = os.path.join(directory, "temp_rename")
    os.makedirs(temp_dir, exist_ok=True)
    
    # First move to temporary files to avoid conflicts
    temp_files = []
    for original_idx, (current_idx, file_path) in enumerate(file_indices):
        temp_filename = f"temp_episode_{original_idx}.hdf5"
        temp_path = os.path.join(temp_dir, temp_filename)
        print(f"Moving {file_path} to temporary file {temp_path}")
        os.rename(file_path, temp_path)
        temp_files.append((original_idx, temp_path))
    
    # Now rename from the temporary files to the final destination
    for new_idx, (_, temp_path) in enumerate(temp_files):
        new_filename = f"episode_{new_idx}.hdf5"
        new_path = os.path.join(directory, new_filename)
        print(f"Renaming to {new_path}")
        os.rename(temp_path, new_path)
    
    # Remove temporary directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        print(f"Note: Could not remove temporary directory {temp_dir}")
    
    print(f"Successfully renamed {len(file_indices)} files to have consecutive indices from 0 to {len(file_indices)-1}")


if __name__ == "__main__":
    # You can specify a directory path as an argument or use the current directory
    rename_episode_files(directory='/media/rby1/T7/dataset/rby1_box_pulling_right_ft')