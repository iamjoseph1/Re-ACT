import socket
import numpy as np
import time
import threading
import logging

class Client2:
    def __init__(self, server_host, server_port, timeout=5):
        """
        Initializes Client2 to receive qpos and send action.

        Args:
            server_host (str): Server IP address.
            server_port_recv (int): Port to receive qpos from (server's recv_port).
            server_port_send (int): Port to send action to (server's send_port).
            timeout (int): Socket timeout in seconds.
        """
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect((self.server_host, self.server_port))
            logging.info(f"Connected to server at {self.server_host}:{self.server_port}")
        except Exception as e:
            logging.error(f"Failed to connect to server: {e}")
            self.sock = None

        self.recv_buffer = ""

    def receive_data(self):
        """
        Receives the qpos data from the server.

        Returns:
            np.ndarray or None: The received qpos array or None if failed.
        """
        if not self.sock:
            logging.error("Socket is not connected.")
            return None
        try:
            while '\n' not in self.recv_buffer:
                data = self.sock.recv(1024).decode('utf-8')
                if not data:
                    # logging.warning("No data received from server.")
                    return None
                self.recv_buffer += data
            data_str, self.recv_buffer = self.recv_buffer.split('\n', 1)
            data = np.array([float(x) for x in data_str.strip().split(',')])
            # logging.info("Received qpos from server.")
            return data
        except Exception as e:
            logging.error(f"Failed to receive qpos: {e}")
            return None
         
        
    def send_action(self, action):
        """
        Sends the action data to the server.

        Args:
            action (np.ndarray): The action data to send.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not self.sock:
            logging.error("Socket is not connected.")
            return False
        try:
            action_str = ','.join([str(x) for x in action]) + '\n'
            self.sock.sendall(action_str.encode('utf-8'))
            # logging.info("Sent action to server.")
            return True
        except Exception as e:
            logging.error(f"Failed to send action: {e}")
            return False
        
    def close(self):
        """Closes sockets."""
        if self.sock:
            self.sock.close()
            logging.info("Closed socket.")
            self.sock = None

    def __del__(self):
        self.close()