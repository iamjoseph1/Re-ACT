import os
import numpy as np
import h5py
from glob import glob
import re
import argparse
import matplotlib.pyplot as plt


class KalmanFilterFT:
    """
    Kalman Filter implementation for Force/Torque data
    """
    def __init__(self, dim=6, Q_scale=1e-3, R_scale=1e-2):
        self.dim = dim  # 6D Force or 6D Torque Data
        
        # State Transition Matrix (A): Identity (Assuming forces change slowly)
        self.A = np.eye(self.dim)
        
        # Measurement Matrix (H): Identity (Direct observation)
        self.H = np.eye(self.dim)
        
        # Process Noise Covariance (Q): Small values allow slow variations
        self.Q = np.eye(self.dim) * Q_scale  
        
        # Measurement Noise Covariance (R): Larger values assume noisy sensor
        self.R = np.eye(self.dim) * R_scale  
        
        # Initial State Estimate (x): Start with zero force/torque
        self.x = np.zeros((self.dim, 1))
        
        # Initial Estimate Covariance (P): Start with high uncertainty
        self.P = np.eye(self.dim)
    
    def update(self, z):
        """Applies one step of the Kalman filter with new F/T measurement z."""
        z = np.reshape(z, (self.dim, 1))  # Convert to column vector
        
        # Prediction Step
        x_pred = self.A @ self.x  
        P_pred = self.A @ self.P @ self.A.T + self.Q  
        
        # Compute Kalman Gain
        S = self.H @ P_pred @ self.H.T + self.R  
        K = P_pred @ self.H.T @ np.linalg.inv(S)  
        
        # Correction Step
        self.x = x_pred + K @ (z - self.H @ x_pred)  
        self.P = (np.eye(self.dim) - K @ self.H) @ P_pred  
        
        return self.x.flatten()  # Return as a 1D array for easy use


class HDF5DataProcessor:
    """
    Class to process HDF5 files, filter force/torque data and save results
    """
    def __init__(self, input_dir, output_dir=None, force_q=1e-3, force_r=1e-2, torque_q=1e-3, torque_r=1e-2):
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.force_q = force_q
        self.force_r = force_r
        self.torque_q = torque_q
        self.torque_r = torque_r
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize arrays for storing data
        self.episode_indices = []
        self.file_paths = []
        
    def find_hdf5_files(self):
        """Find all episode HDF5 files in the input directory"""
        file_pattern = os.path.join(self.input_dir, 'episode_*.hdf5')
        all_files = glob(file_pattern)
        
        # Filter out already processed files
        self.file_paths = [f for f in all_files if '_filtered.hdf5' not in f]
        
        # Extract indices from filenames
        for file_path in self.file_paths:
            match = re.search(r'episode_(\d+)\.hdf5', os.path.basename(file_path))
            if match:
                self.episode_indices.append(int(match.group(1)))
            else:
                self.file_paths.remove(file_path)  # Remove files that don't match the pattern
        
        print(f"Found {len(self.file_paths)} HDF5 files to process")
        return len(self.file_paths) > 0
    
    def load_ft_data(self, hdf5_file):
        """Load force and torque data from HDF5 file"""
        with h5py.File(hdf5_file, 'r') as f:
            # Extract force/torque data from the observations group
            force_data = f['/observations/force'][:]
            torque_data = f['/observations/torque'][:]
            
        return force_data, torque_data
    
    def process_files(self, visualize=False, vis_dir=None):
        """Process all found HDF5 files"""
        if not self.find_hdf5_files():
            print("No HDF5 files found to process")
            return
        
        if visualize and vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
        
        for idx, (file_idx, file_path) in enumerate(zip(self.episode_indices, self.file_paths)):
            print(f"Processing file {idx+1}/{len(self.file_paths)}: episode_{file_idx}.hdf5")
            
            try:
                # Load force and torque data
                force_data, torque_data = self.load_ft_data(file_path)
                
                # Get dimensions of force and torque data
                force_dim = force_data.shape[1]
                torque_dim = torque_data.shape[1]
                
                # Initialize Kalman Filters with specified parameters
                kf_f = KalmanFilterFT(dim=force_dim, Q_scale=self.force_q, R_scale=self.force_r)
                kf_t = KalmanFilterFT(dim=torque_dim, Q_scale=self.torque_q, R_scale=self.torque_r)
                
                # Apply filters
                filtered_force = np.array([kf_f.update(f) for f in force_data])
                filtered_torque = np.array([kf_t.update(t) for t in torque_data])
                
                # Create output filename
                output_file = os.path.join(self.output_dir, f'episode_{file_idx}_filtered.hdf5')
                
                # Save filtered data to new file
                self.create_filtered_file(file_path, output_file, filtered_force, filtered_torque)
                
                # Visualize results if requested
                if visualize and vis_dir:
                    self.visualize_filtering(file_idx, force_data, filtered_force, torque_data, filtered_torque, vis_dir)
                    
                print(f"✓ Successfully processed episode_{file_idx}.hdf5")
            
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    def create_filtered_file(self, source_path, destination_path, filtered_force, filtered_torque):
        """Create a new HDF5 file with filtered force/torque data"""
        with h5py.File(source_path, "r") as src_file:
            with h5py.File(destination_path, "w") as dst_file:
                # Copy all groups and datasets from source to destination
                for key in src_file.keys():
                    if key != "observations":
                        # Direct copy for non-observations groups
                        src_file.copy(key, dst_file)
                    else:
                        # Create observations group in destination
                        obs_group = dst_file.create_group("observations")
                        
                        # Copy all datasets in observations except force and torque
                        for obs_key in src_file["observations"].keys():
                            if obs_key != "force" and obs_key != "torque":
                                src_file["observations"].copy(obs_key, obs_group)
                        
                        # Create new force and torque datasets with filtered data
                        obs_group.create_dataset("force", data=filtered_force)
                        obs_group.create_dataset("torque", data=filtered_torque)
    
    def visualize_filtering(self, episode_idx, orig_force, filt_force, orig_torque, filt_torque, vis_dir):
        """Create visualizations of original vs filtered data"""
        # Plot force
        plt.figure(figsize=(12, 8))
        for i in range(min(orig_force.shape[1], 6)):
            plt.subplot(3, 2, i+1)
            plt.plot(orig_force[:, i], 'b-', alpha=0.5, label='Original')
            plt.plot(filt_force[:, i], 'r-', label='Filtered')
            plt.title(f'Force Component {i+1}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'episode{episode_idx}_force.png'))
        plt.close()
        
        # Plot torque
        plt.figure(figsize=(12, 8))
        for i in range(min(orig_torque.shape[1], 6)):
            plt.subplot(3, 2, i+1)
            plt.plot(orig_torque[:, i], 'b-', alpha=0.5, label='Original')
            plt.plot(filt_torque[:, i], 'r-', label='Filtered')
            plt.title(f'Torque Component {i+1}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'episode_{episode_idx}_torque.png'))
        plt.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process HDF5 files with Kalman filtering for force/torque data')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing episode*.hdf5 files')
    parser.add_argument('--output_dir', type=str, help='Directory to save filtered HDF5 files (defaults to input_dir)')
    parser.add_argument('--force_q', type=float, default=1e-3, help='Process noise scale for force filter')
    parser.add_argument('--force_r', type=float, default=0.5*1e-1, help='Measurement noise scale for force filter')
    parser.add_argument('--torque_q', type=float, default=1e-3, help='Process noise scale for torque filter')
    parser.add_argument('--torque_r', type=float, default=1e-2, help='Measurement noise scale for torque filter')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("HDF5 Force/Torque Data Processor")
    print("-" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir or args.input_dir}")
    print(f"Force filter parameters: Q={args.force_q}, R={args.force_r}")
    print(f"Torque filter parameters: Q={args.torque_q}, R={args.torque_r}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    if args.visualize:
        print(f"Visualization directory: {args.vis_dir}")
    print("-" * 40)
    
    # Create processor and run
    processor = HDF5DataProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force_q=args.force_q,
        force_r=args.force_r,
        torque_q=args.torque_q,
        torque_r=args.torque_r
    )
    
    processor.process_files(visualize=args.visualize, vis_dir=args.vis_dir)
    
    print("Processing complete!")