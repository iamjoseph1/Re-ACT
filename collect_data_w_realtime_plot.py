import socket
import threading
import time
import h5py
import numpy as np
import torch
import pyrealsense2 as rs
import pickle
import sys
import os
import signal
import queue
import matplotlib.pyplot as plt
# filtering
from kalman_filter import KalmanFilterFT

class RealSenseHandler:
    def __init__(self, serial_numbers, mode='collect'):
        """
        Initializes the RealSenseHandler with camera serial numbers.
        """
        self.serial_numbers = serial_numbers
        self.pipelines = []
        self.running = True
        self.lock = threading.Lock()  # 최신 프레임 업데이트를 위한 락
        self.mode = mode

        for serial in self.serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            try:
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
                pipeline.start(config)
                self.pipelines.append(pipeline)
                print(f"[INFO] Camera with serial {serial} initialized successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to initialize camera with serial {serial}: {e}")

        self.latest_frames = None  # (cam1, cam2, cam3)
        self.fetch_thread = None  # 프레임 캡처 스레드

    def crop_center(self, frame, target_w=640, target_h=480):
        """
        - frame.shape: (H, W, C)
        """
        H, W, C = frame.shape
        
        center_x = W // 2  
        center_y = H // 2  

        start_x = center_x - (target_w // 2)
        end_x   = center_x + (target_w // 2)
        start_y = center_y - (target_h // 2)
        end_y   = center_y + (target_h // 2)

        if start_x < 0: start_x = 0
        if start_y < 0: start_y = 0
        if end_x > W: end_x = W
        if end_y > H: end_y = H

        cropped = frame[start_y:end_y, start_x:end_x, :]

        return cropped


    def get_frames(self):
        """
        Waits for frames from each camera and returns (cam1_frame, cam2_frame, cam3_frame, timestamp).
        """
        frames_out = []
        for idx, pipeline in enumerate(self.pipelines):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame_data = np.asanyarray(color_frame.get_data()).astype(np.uint8)
                cropped_frame = self.crop_center(frame_data, 640, 480)
                frames_out.append(cropped_frame)
            else:
                print(f"[WARNING] No color frame received from camera {idx}")
                frames_out.append(None)

        while len(frames_out) < 3:
            print("[WARNING] Checking the camera number !")
            frames_out.append(None)

        timestamp = time.time()

        if self.mode == 'collect':
            return ((timestamp, frames_out[0]), 
                    (timestamp, frames_out[1]), 
                    (timestamp, frames_out[2]))
        else:
            return frames_out[0], frames_out[1], frames_out[2]
        # return frames_out[0], frames_out[1], frames_out[2]
        # return (timestamp, frames_out[0]), (timestamp, frames_out[1]), (timestamp, frames_out[2])

    def start_frame_fetcher(self):
        self.fetch_thread = threading.Thread(target=self._frame_fetcher, daemon=True)
        self.fetch_thread.start()

    def _frame_fetcher(self):
        while self.running:
            frames = self.get_frames()  # 동기적으로 프레임 캡처
            with self.lock:
                self.latest_frames = frames


    def get_latest_frames(self):
        """
        return : (cam1_frame, cam2_frame, cam3_frame)
        """
        with self.lock:
            return self.latest_frames

    def stop(self):
        self.running = False
        if self.fetch_thread is not None:
            self.fetch_thread.join()
        for pipeline in self.pipelines:
            pipeline.stop()
        print("[INFO] RealSense cameras stopped.")


class BufferManager:
    def __init__(self, max_steps=400):
        """
        Manages synchronized data storage (action, qpos, frames).
        """
        self.synced_data = []  # List to store (action, qpos, ft, frame1, frame2, frame3)
        self.lock = threading.Lock()
        self.max_steps = max_steps

    def add_synced_data(self, action, qpos, force, torque, cam1_frame, cam2_frame, cam3_frame):
        """
        Adds a set of data to the synced_data list.
        """
        with self.lock:
            self.synced_data.append((action, qpos, force, torque, cam1_frame, cam2_frame, cam3_frame))


    def reset(self):
        """
        Clears the current buffer.
        """
        with self.lock:
            self.synced_data = []
        print("[INFO] Buffer has been reset.")


    def interpolate_synced_data(synced_data):
        """
        synced_data: 샘플들의 리스트, 각 샘플은 아래와 같은 튜플:
            (action, qpos, force, torque, cam1, cam2, cam3)
        where:
            action: (action_ts, action_vector)
            qpos:   (qpos_ts, qpos_vector)
            force:   (force_ts, force_vector)
            torque:   (torque_ts, torque_vector)
            camX:   (timestamp, frame)
        
        Returns:
            interp_actions: (N, d_action) numpy array - 카메라 타임스탬프 기준으로 보간된 액션 데이터
            interp_qpos:   (N, d_qpos) numpy array - 카메라 타임스탬프 기준으로 보간된 qpos 데이터
            interp_force:   (N, d_force) numpy array - 카메라 타임스탬프 기준으로 보간된 force 데이터
            interp_torque:   (N, d_torque) numpy array - 카메라 타임스탬프 기준으로 보간된 torque 데이터
            front_frames:  list of front frames (cam3), 길이 N
            left_frames:   list of left frames (cam1), 길이 N
            right_frames:  list of right frames (cam2), 길이 N
            camera_ts:     (N,) numpy array of camera timestamps (cam1의 타임스탬프 사용)
        """


        camera_ts = []
        action_ts = []
        action_values = []
        qpos_values = []
        force_values = []
        torque_values = []
        left_frames = []   # cam1
        right_frames = []  # cam2
        front_frames = []  # cam3

        for sample in synced_data:

            action, qpos, force, torque, cam1, cam2, cam3 = sample
            # 카메라 타임스탬프는 cam1에서 추출 (모든 카메라의 timestamp가 같다고 가정)
            cam_timestamp = cam1[0]
            camera_ts.append(cam_timestamp)
            action_ts.append(action[0])
            action_values.append(action[1])
            qpos_values.append(qpos[1])
            force_values.append(force[1])
            torque_values.append(torque[1])
            left_frames.append(cam1[1])
            right_frames.append(cam2[1])
            front_frames.append(cam3[1])
        
        camera_ts = np.array(camera_ts)
        action_ts = np.array(action_ts)
        action_values = np.array(action_values)  # shape: (N, d_action)
        qpos_values = np.array(qpos_values)      # shape: (N, d_qpos)
        force_values = np.array(force_values)      # shape: (N, d_ft)
        torque_values = np.array(torque_values)      # shape: (N, d_ft)

        N = len(camera_ts)
        d_action = action_values.shape[1]
        d_qpos = qpos_values.shape[1]
        d_force = force_values.shape[1]
        d_torque = torque_values.shape[1]

        interp_actions = np.zeros((N, d_action))
        interp_qpos = np.zeros((N, d_qpos))
        interp_force = np.zeros((N, d_force))
        interp_torque = np.zeros((N, d_torque))

        # 각 채널별로 선형 보간 진행 (np.interp는 1차원 데이터 보간 함수)
        for j in range(d_action):
            interp_actions[:, j] = np.interp(camera_ts, action_ts, action_values[:, j])
        for j in range(d_qpos):
            interp_qpos[:, j] = np.interp(camera_ts, action_ts, qpos_values[:, j])
        for j in range(d_force):
            interp_force[:, j] = np.interp(camera_ts, action_ts, force_values[:, j])
        for j in range(d_torque):
            interp_torque[:, j] = np.interp(camera_ts, action_ts, torque_values[:, j])
        
        return interp_actions, interp_qpos, interp_force, interp_torque, front_frames, left_frames, right_frames



    def save_to_pickle(self, filename):
        """
        Saves collected data as tensors directly to a pickle file.
        Resets the buffer after saving.
        The saved data is organized as:
        {
            'observation': {
                'images': {
                    'front': (max_steps, 480, 640, 3),
                    'left':  (max_steps, 480, 640, 3),
                    'right': (max_steps, 480, 640, 3)
                },
                'qpos': (max_steps, 16)
            },
            'action': (max_steps, 16)
        }
        """
        with self.lock:
            if len(self.synced_data) == 0:
                print(f"[WARNING] No data available to save.")
                return


            # Save exactly max_steps
            if len(self.synced_data) < self.max_steps:
                print(f"[WARNING] Not enough data to save. Current steps: {len(self.synced_data)}. Required: {self.max_steps}.")
                return

        data_to_process = self.synced_data[:self.max_steps]
        interp_actions, interp_qpos, front_frames, left_frames, right_frames = BufferManager.interpolate_synced_data(data_to_process)

        # Convert the lists of frames into NumPy arrays.
        front_array = np.stack(front_frames)   # Expected shape: (max_steps, 480, 640, 3)
        left_array  = np.stack(left_frames)      # Expected shape: (max_steps, 480, 640, 3)
        right_array = np.stack(right_frames)     # Expected shape: (max_steps, 480, 640, 3)

        # Build the data structure.
        data_to_save = {
            'observations': {
                'images': {
                    'front': front_array,
                    'left':  left_array,
                    'right': right_array,
                },
                'qpos': interp_qpos
            },
            'action': interp_actions
        }


        try:
            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"[INFO] Synchronized data saved to {filename} as a pickle file.")
        except Exception as e:
            print(f"[ERROR] Failed to save data to {filename}: {e}")

    def save_to_hdf5(self, filename):

        with self.lock:
            if len(self.synced_data) == 0:
                print(f"[WARNING] No data available to save.")
                return
            if len(self.synced_data) < self.max_steps:
                print(f"[WARNING] Not enough data to save. Current steps: {len(self.synced_data)}. Required: {self.max_steps}.")
                return

        data_to_process = self.synced_data[:self.max_steps]
        interp_actions, interp_qpos, interp_force, interp_torque, front_frames, left_frames, right_frames = BufferManager.interpolate_synced_data(data_to_process)

        front_array = np.stack(front_frames)  # (max_steps, 480, 640, 3)
        left_array  = np.stack(left_frames)   # (max_steps, 480, 640, 3)
        right_array = np.stack(right_frames)  # (max_steps, 480, 640, 3)

        # # in case of using filtering
        # kf_force = KalmanFilterFT()
        # kf_torque = KalmanFilterFT()

        # # Apply filters
        # interp_force = np.array([kf_force.update(f) for f in interp_force])
        # interp_torque = np.array([kf_torque.update(t) for t in interp_torque])

        try:
            with h5py.File(filename, 'w') as h5f:
                obs_group = h5f.create_group('observations')
                
                img_group = obs_group.create_group('images')
            
                img_group.create_dataset('front', data=front_array)
                img_group.create_dataset('left', data=left_array)
                img_group.create_dataset('right', data=right_array)

                obs_group.create_dataset('qpos', data=interp_qpos, dtype=np.float32)
                obs_group.create_dataset('force', data=interp_force, dtype=np.float32)
                obs_group.create_dataset('torque', data=interp_torque, dtype=np.float32)

                h5f.create_dataset('action', data=interp_actions, dtype=np.float32)

            print(f"[INFO] Synchronized data saved to {filename} as a HDF5 file.")
        except Exception as e:
            print(f"[ERROR] Failed to save data to {filename}: {e}")

class SaveCounter:
    def __init__(self, initial=1):
        self.count = initial
        self.lock = threading.Lock()
    
    def get_count(self):
        with self.lock:
            return self.count
    
    def increment(self):
        with self.lock:
            current = self.count
            self.count += 1
            return current

def process_received_data(message):
    """
    Converts received bytes into a torch.Tensor (e.g., split by commas).
    """
    try:
        state = np.array(list(map(float, message.strip().split(','))))
        return state
    except ValueError as ve:
        print(f"[ERROR] ValueError while processing message: {message}")
        raise ve

# Add this function to handle real-time plotting
def plot_force_magnitude(force_queue, stop_event):
    """
    Real-time plot of the magnitude of current_force.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Force Magnitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Magnitude (L2 Norm)")
    ax.grid(True)

    x_data, y_data = [], []
    start_time = time.time()

    line, = ax.plot(x_data, y_data, label="Force Magnitude")
    ax.legend()

    while not stop_event.is_set():
        try:
            force_magnitude = force_queue.get(timeout=0.1)
            elapsed_time = time.time() - start_time

            x_data.append(elapsed_time)
            y_data.append(force_magnitude)

            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.relim()
            ax.autoscale_view()

            plt.pause(0.001)  # Pause to update the plot
        except queue.Empty:
            continue  # No new data, continue looping

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Modify the start_server function to calculate and send force magnitude to the queue
def start_server(host, port, realsense_handler, buffer_manager, stop_event, save_queue, max_steps, buffer_available, save_requested_event, receive_event, force_queue):
    """
    Starts a TCP server.
    Whenever a state is received, immediately fetch cam1, cam2 & cam3 frames and store (action, qpos, frames).
    If the connection is closed by the sender or stop_event is set, save to pickle and stop.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"[INFO] Server listening on {host}:{port}")

        server_socket.settimeout(1.0)  # Set timeout to check for stop_event

        client_socket, client_address = server_socket.accept()
        print(f"[INFO] Connection established with {client_address}")

        kf_force = KalmanFilterFT()
        kf_torque = KalmanFilterFT()

        with client_socket:
            data_buffer = ""
            prev_action = None

            while not stop_event.is_set():
                try:
                    data = client_socket.recv(4096)

                    if not data:
                        print("[INFO] Sender closed the connection.")
                        break

                    data_buffer += data.decode('utf-8')

                    # Split messages by '\n'
                    while '\n' in data_buffer:
                        message, data_buffer = data_buffer.split('\n', 1)
                        if message:
                            try:
                                state = process_received_data(message).copy()
                            except ValueError:
                                print(f"[WARNING] Skipping malformed message: {message}")
                                continue

                            current_action = (state[0], state[1:17])
                            current_qpos = (state[0], state[17:33])
                            kf_force_value = kf_force.update(state[33:39])
                            kf_torque_value = kf_torque.update(state[39:])
                            current_force = (state[0], kf_force_value)
                            current_torque = (state[0], kf_torque_value)

                            force_magnitude = np.linalg.norm(kf_force_value[3:])  # Calculate L2 norm of right arm force

                            # Send force magnitude to the plotting queue
                            # force_queue.put(force_magnitude)
                            force_queue.put(kf_force_value[3])

                            if prev_action is not None and buffer_available.is_set():
                                cam1_frame, cam2_frame, cam3_frame = realsense_handler.get_latest_frames()
                                if cam1_frame is None or cam2_frame is None or cam3_frame is None:
                                    print("[WARNING] Failed to fetch all frames. Skipping this step.")
                                    continue

                                buffer_manager.add_synced_data(prev_action, current_qpos, current_force, current_torque, cam1_frame, cam2_frame, cam3_frame)

                                with buffer_manager.lock:
                                    current_steps = len(buffer_manager.synced_data)

                                if current_steps == max_steps:
                                    print(f"[INFO] Buffer has reached {max_steps} steps.")
                                    if not save_requested_event.is_set():
                                        print("[INFO] Triggering auto_save.")
                                        save_queue.put("auto_save")
                                        buffer_available.clear()
                                    else:
                                        pass

                            prev_action = current_action

                except socket.timeout:
                    continue  # Check stop_event

                except Exception as e:
                    print(f"[ERROR] Error while receiving data: {e}")
                    break

    finally:
        server_socket.close()
        print("[INFO] Server socket closed.")

def save_monitor(buffer_manager, stop_event, save_queue, save_counter, demo_dir, prompt_event, buffer_available, save_requested_event):
    """
    Monitors for a save request and saves data when buffer reaches max_steps.
    Notifies the main thread to prompt the user after saving.
    """
    saving_lock = threading.Lock()
    while not stop_event.is_set():
        try:
            # Wait for a save signal
            message = save_queue.get(timeout=1)  # Wait for 1 second
            with saving_lock:
                if message == "auto_save":
                    # Check again if a save has been requested to avoid duplicate saves
                    if not save_requested_event.is_set():
                        # Save 400 steps
                        filename = os.path.join(demo_dir, f"episode_{save_counter.get_count()-1}.hdf5")
                        buffer_manager.save_to_hdf5(filename)

                        # Increment save count
                        save_counter.increment()

                        # Notify main thread to prompt user
                        prompt_event.set()
                    else:
                        print("[INFO] Save has already been requested. Skipping auto_save.")

                elif message == "requested_save":
                    print("[INFO] Save requested. Waiting until buffer reaches 400 steps.")
                    # Indicate that a save has been requested
                    save_requested_event.set()
                    # Stop adding more data
                    buffer_available.clear()

                    while not stop_event.is_set():
                        with buffer_manager.lock:
                            current_steps = len(buffer_manager.synced_data)
                            last_data = buffer_manager.synced_data[-1]


                        # Calculate how many duplicates are needed
                        duplicates_needed = buffer_manager.max_steps - current_steps
                        duplicates_needed = max(duplicates_needed, 0)

                        # Add duplicates to fill the buffer
                        for _ in range(duplicates_needed):
                            buffer_manager.add_synced_data(*last_data)

                        if len(buffer_manager.synced_data) == buffer_manager.max_steps:
                            # Save 400 steps
                            filename = os.path.join(demo_dir, f"episode_{save_counter.get_count()-1}.hdf5")
                            buffer_manager.save_to_hdf5(filename)

                            # Increment save count
                            save_counter.increment()
                            # Notify main thread to prompt user
                            prompt_event.set()
                            # Clear the save request flag
                            save_requested_event.clear()
                            break

                        else:
                            pass


        except queue.Empty:
            continue  # Continue checking stop_event

def signal_handler_factory(save_queue, save_requested_event):
    """
    Creates a signal handler that signals the server to save data.
    """
    def signal_handler(sig, frame):
        """
        Handles Ctrl + Z signal.
        """
        print("\n[INFO] Ctrl + Z detected. Initiating save process when buffer reaches 400 steps...")
        # Only put "requested_save" if a save hasn't been requested already
        if not save_requested_event.is_set():
            save_queue.put("requested_save")
        else:
            pass

    return signal_handler

# Modify the main function to start the plotting thread
def main():
    global data_buffer
    global prev_action

    serial_numbers = ["f0210565", "f1380687", "f0244932"]  # left, right, front
    realsense_handler = RealSenseHandler(serial_numbers, mode='collect')
    realsense_handler.start_frame_fetcher()

    buffer_manager = BufferManager(max_steps=400)

    stop_event = threading.Event()
    save_queue = queue.Queue()
    prompt_event = threading.Event()
    receive_event = threading.Event()

    save_counter = SaveCounter(initial=1)
    demo_dir = "/media/rby1/T7/dataset/rby1_box_wiping_ft"
    os.makedirs(demo_dir, exist_ok=True)

    buffer_available = threading.Event()
    buffer_available.set()

    save_requested_event = threading.Event()

    # Create a queue for force magnitude
    force_queue = queue.Queue()

    # Start the plotting thread
    plot_thread = threading.Thread(
        target=plot_force_magnitude,
        args=(force_queue, stop_event),
        daemon=True
    )
    plot_thread.start()

    server_thread = threading.Thread(
        target=start_server,
        args=(
            "192.168.200.169",
            8000,
            realsense_handler,
            buffer_manager,
            stop_event,
            save_queue,
            buffer_manager.max_steps,
            buffer_available,
            save_requested_event,
            receive_event,
            force_queue  # Pass the force queue to the server
        ),
        daemon=True
    )
    server_thread.start()

    save_monitor_thread = threading.Thread(
        target=save_monitor,
        args=(
            buffer_manager,
            stop_event,
            save_queue,
            save_counter,
            demo_dir,
            prompt_event,
            buffer_available,
            save_requested_event
        ),
        daemon=True
    )
    save_monitor_thread.start()

    signal.signal(signal.SIGTSTP, signal_handler_factory(save_queue, save_requested_event))

    try:
        while not stop_event.is_set():
            if prompt_event.is_set():
                while True:
                    answer = input("Do you want to continue? (yes/no): ").strip().lower()
                    if answer == "yes":
                        data_buffer = ""
                        prev_action = None
                        buffer_manager.reset()
                        print("[INFO] Continuing data collection. Press Ctrl + Z again to save data.")
                        prompt_event.clear()
                        receive_event.set()
                        buffer_available.set()
                        break
                    elif answer == "no":
                        print("[INFO] Exiting program...")
                        stop_event.set()
                        save_monitor_thread.join()
                        realsense_handler.stop()
                        sys.exit()
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
    except KeyboardInterrupt:
        sys.exit()

    print("[INFO] Program exited successfully.")

if __name__ == "__main__":
    main()
# when the task is done, the user realease the button then the received state be equal always