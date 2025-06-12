import numpy as np

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