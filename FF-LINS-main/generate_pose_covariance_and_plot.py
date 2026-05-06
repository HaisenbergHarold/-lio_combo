#!/usr/bin/env python3
"""
Generate pose covariance from FF-LINS trajectory and plot 3-sigma curves.
This script can:
1. Load trajectory covariance from C++ generated trajectory_covariance.txt (if exists)
2. Or compute covariance estimates from trajectory.csv characteristics
3. Plot 3σ curves for x, y, z, and yaw (four subplots)

Supports two input formats:
- Format 1 (C++ output): timestamp, pos_xyz_3sigma, quat_xyzw_3sigma, vel_xyz_3sigma
- Format 2 (Python output): timestamp, pos_xyz_3sigma, rpy_3sigma
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def quaternion_to_euler_batch(qx, qy, qz, qw):
    """Convert quaternions to roll, pitch, yaw Euler angles (in radians)."""
    n = len(qx)
    roll = np.zeros(n)
    pitch = np.zeros(n)
    yaw = np.zeros(n)
    
    for i in range(n):
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw[i] * qx[i] + qy[i] * qz[i])
        cosr_cosp = 1.0 - 2.0 * (qx[i] * qx[i] + qy[i] * qy[i])
        roll[i] = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw[i] * qy[i] - qz[i] * qx[i])
        if abs(sinp) >= 1:
            pitch[i] = math.copysign(math.pi / 2, sinp)
        else:
            pitch[i] = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw[i] * qz[i] + qx[i] * qy[i])
        cosy_cosp = 1.0 - 2.0 * (qy[i] * qy[i] + qz[i] * qz[i])
        yaw[i] = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def quaternion_variance_to_yaw_sigma(qx_var, qy_var, qz_var, qw_var, qx, qy, qz, qw):
    """
    Convert quaternion covariance to yaw angle standard deviation.
    Uses error propagation from quaternion space to Euler angle space.
    
    Args:
        qx_var, qy_var, qz_var, qw_var: quaternion variance (diagonal of covariance)
        qx, qy, qz, qw: mean quaternion values
    
    Returns:
        yaw_std: standard deviation of yaw angle in radians
    """
    # Compute partial derivatives of yaw w.r.t. quaternion components
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    
    denom = 1.0 - 2.0 * (qy**2 + qz**2)
    numer = 2.0 * (qw*qz + qx*qy)
    
    # Avoid division by zero
    if abs(denom) < 1e-10 or (numer**2 + denom**2) < 1e-20:
        return 0.001  # Default small value
    
    # Partial derivatives (simplified Jacobian)
    d_yaw_d_qx = 2.0 * qy / (numer**2 + denom**2) * denom
    d_yaw_d_qy = 2.0 * (qx * denom + 2.0 * qy * numer) / (numer**2 + denom**2)
    d_yaw_d_qz = 2.0 * (qw * denom + 2.0 * qz * numer) / (numer**2 + denom**2)
    d_yaw_d_qw = 2.0 * qz / (numer**2 + denom**2) * denom
    
    # Error propagation: sigma_yaw^2 = J * Cov_q * J^T
    yaw_var = (d_yaw_d_qx**2 * qx_var + 
               d_yaw_d_qy**2 * qy_var + 
               d_yaw_d_qz**2 * qz_var + 
               d_yaw_d_qw**2 * qw_var)
    
    yaw_std = math.sqrt(max(yaw_var, 1e-12))
    return yaw_std

def load_cpp_covariance(filename):
    """
    Load covariance data from C++ generated trajectory_covariance.txt.
    Format: timestamp pos_x_3sigma pos_y_3sigma pos_z_3sigma quat_x_3sigma quat_y_3sigma quat_z_3sigma quat_w_3sigma vel_x_3sigma vel_y_3sigma vel_z_3sigma
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 11:
                try:
                    row = [float(x) for x in parts[:11]]
                    data.append(row)
                except ValueError:
                    continue
    
    if len(data) == 0:
        return None
    
    cov_array = np.array(data)
    
    # Sort by timestamp and remove duplicates
    if len(cov_array) > 0:
        # Sort by timestamp (first column)
        sorted_indices = np.argsort(cov_array[:, 0])
        sorted_array = cov_array[sorted_indices]
        
        # Remove duplicate timestamps, keep first occurrence
        unique_times, unique_indices = np.unique(sorted_array[:, 0], return_index=True)
        cov_array = sorted_array[unique_indices]
        
        print(f"Sorted and removed duplicates: {len(sorted_array)} -> {len(cov_array)} points")
    
    # Extract position 3-sigma (already 3-sigma from C++)
    times = cov_array[:, 0]
    pos_x_3sigma = cov_array[:, 1]
    pos_y_3sigma = cov_array[:, 2]
    pos_z_3sigma = cov_array[:, 3]
    
    # Quaternion 3-sigma (already 3-sigma from C++)
    quat_x_3sigma = cov_array[:, 4]
    quat_y_3sigma = cov_array[:, 5]
    quat_z_3sigma = cov_array[:, 6]
    quat_w_3sigma = cov_array[:, 7]
    
    # Convert quaternion 3-sigma to yaw 3-sigma
    # Note: The C++ code outputs quaternion component uncertainties directly
    # We need to convert these to yaw uncertainty
    yaw_3sigma = np.zeros(len(times))
    for i in range(len(times)):
        # Approximate conversion: use the dominant quaternion component variance
        # This is a simplified approach; for better accuracy, need full Jacobian
        yaw_3sigma[i] = 3.0 * math.sqrt(
            (quat_x_3sigma[i]/3.0)**2 * 0.25 + 
            (quat_y_3sigma[i]/3.0)**2 * 0.25 + 
            (quat_z_3sigma[i]/3.0)**2 * 0.25 + 
            (quat_w_3sigma[i]/3.0)**2 * 0.25
        )
    
    # Combine into output format: [time, pos_x_3sigma, pos_y_3sigma, pos_z_3sigma, yaw_3sigma]
    result = np.column_stack([times, pos_x_3sigma, pos_y_3sigma, pos_z_3sigma, yaw_3sigma])
    
    return result

def load_python_covariance(filename):
    """
    Load covariance data from Python generated file.
    Format: timestamp pos_x_3sigma pos_y_3sigma pos_z_3sigma roll_3sigma pitch_3sigma yaw_3sigma
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    row = [float(x) for x in parts[:7]]
                    data.append(row)
                except ValueError:
                    continue
    
    if len(data) == 0:
        return None
    
    cov_array = np.array(data)
    
    # Sort by timestamp and remove duplicates (similar to C++ covariance loading)
    if len(cov_array) > 0:
        # Sort by timestamp (first column)
        sorted_indices = np.argsort(cov_array[:, 0])
        sorted_array = cov_array[sorted_indices]
        
        # Remove duplicate timestamps, keep first occurrence
        unique_times, unique_indices = np.unique(sorted_array[:, 0], return_index=True)
        cov_array = sorted_array[unique_indices]
        
        print(f"Sorted and removed duplicates: {len(sorted_array)} -> {len(cov_array)} points")
    
    # Extract relevant columns: [time, pos_x_3sigma, pos_y_3sigma, pos_z_3sigma, yaw_3sigma]
    result = cov_array[:, [0, 1, 2, 3, 6]]
    
    return result

def compute_trajectory_covariance(times, positions, euler_angles):
    """
    Compute covariance estimates from trajectory data.
    
    Args:
        times: timestamps (seconds)
        positions: Nx3 array of (x, y, z) positions
        euler_angles: Nx3 array of (roll, pitch, yaw) in radians
        
    Returns:
        covariance_data: Nx5 array [timestamp, pos_x_3sigma, pos_y_3sigma, pos_z_3sigma, yaw_3sigma]
    """
    n = len(times)
    
    # Initialize covariance arrays
    pos_cov = np.zeros((n, 3))  # Position 3σ (x, y, z)
    yaw_cov = np.zeros(n)       # Yaw 3σ
    
    # Window size for local analysis (1 second window at ~10Hz)
    window_size = min(11, n // 10)
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    
    for i in range(n):
        # Determine window indices
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Window data
        window_times = times[start:end]
        window_pos = positions[start:end]
        window_yaw = euler_angles[start:end, 2]  # Only yaw
        
        if len(window_times) < 3:
            # Not enough data, use default small values
            pos_cov[i] = [0.001, 0.001, 0.002]  # 0.001 m, 0.001 m, 0.002 m
            yaw_cov[i] = 0.002  # ~0.002 rad (~0.114°)
            continue
        
        # Compute position variability using linear fit residuals
        for axis in range(3):
            pos_vals = window_pos[:, axis]
            A = np.vstack([np.ones_like(window_times), window_times]).T
            try:
                coeff, residuals, rank, s = np.linalg.lstsq(A, pos_vals, rcond=None)
                if len(residuals) > 0:
                    std_residual = np.sqrt(residuals[0] / len(window_times))
                    pos_cov[i, axis] = 3.0 * std_residual
                else:
                    pos_cov[i, axis] = 3.0 * np.std(pos_vals)
            except:
                pos_cov[i, axis] = 3.0 * np.std(pos_vals)
        
        # Compute yaw variability
        yaw_mean = np.mean(window_yaw)
        yaw_diff = window_yaw - yaw_mean
        # Wrap differences to [-pi, pi]
        yaw_diff = np.remainder(yaw_diff + np.pi, 2*np.pi) - np.pi
        yaw_std = np.std(yaw_diff)
        yaw_cov[i] = 3.0 * yaw_std
    
    # Apply smoothing and constraints
    pos_cov = np.maximum(pos_cov, 0.0005)  # At least 0.5 mm
    pos_cov = np.minimum(pos_cov, 0.1)     # At most 10 cm
    
    yaw_cov = np.maximum(yaw_cov, 0.0001)  # At least ~0.0057° (≅ 0.0001 rad)
    yaw_cov = np.minimum(yaw_cov, 0.1)     # At most ~5.73° (≅ 0.1 rad)
    
    # Combine into final covariance data
    covariance_data = np.column_stack([
        times,
        pos_cov[:, 0],  # pos_x_3sigma
        pos_cov[:, 1],  # pos_y_3sigma
        pos_cov[:, 2],  # pos_z_3sigma
        yaw_cov         # yaw_3sigma
    ])
    
    return covariance_data

def load_trajectory(filename):
    """Load trajectory data from trajectory.csv."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    row = [float(x) for x in parts[:8]]
                    data.append(row)
                except ValueError:
                    print(f"Warning: Skipping malformed line: {line}")
    return np.array(data)

def save_covariance_to_txt(covariance_data, output_file):
    """Save covariance data to text file."""
    with open(output_file, 'w') as f:
        f.write("# Pose 3-Sigma Standard Deviations\n")
        f.write("# timestamp(seconds), pos_x_3sigma(m), pos_y_3sigma(m), pos_z_3sigma(m), yaw_3sigma(deg)\n")
        
        for row in covariance_data:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")
    
    print(f"Covariance data saved to {output_file}")

def plot_3sigma_curves(covariance_data, trajectory_data=None, output_file=None):
    """Plot 3-sigma curves for x, y, z, and yaw."""
    if len(covariance_data) == 0:
        print("Error: No covariance data to plot")
        return
    
    # Extract data
    times = covariance_data[:, 0]
    pos_x_sigma = covariance_data[:, 1]  # pos_x_3sigma
    pos_y_sigma = covariance_data[:, 2]  # pos_y_3sigma
    pos_z_sigma = covariance_data[:, 3]  # pos_z_3sigma
    yaw_sigma = covariance_data[:, 4]    # yaw_3sigma
    
    # Extract yaw from trajectory for reference
    yaw_deg = None
    if trajectory_data is not None and len(trajectory_data) > 0:
        qx, qy, qz, qw = trajectory_data[:,4], trajectory_data[:,5], trajectory_data[:,6], trajectory_data[:,7]
        roll, pitch, yaw = quaternion_to_euler_batch(qx, qy, qz, qw)
        yaw_deg = yaw * 180 / math.pi
    
    # Create figure with 4 subplots (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot X position 3σ
    ax1.plot(times, pos_x_sigma, 'b-', linewidth=2.0, label='X 3σ (m)')
    ax1.fill_between(times, 0, pos_x_sigma, alpha=0.3, color='b')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Position X 3σ (m)', fontsize=12)
    ax1.set_title('X Position Uncertainty (3σ)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot Y position 3σ
    ax2.plot(times, pos_y_sigma, 'g-', linewidth=2.0, label='Y 3σ (m)')
    ax2.fill_between(times, 0, pos_y_sigma, alpha=0.3, color='g')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Position Y 3σ (m)', fontsize=12)
    ax2.set_title('Y Position Uncertainty (3σ)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot Z position 3σ
    ax3.plot(times, pos_z_sigma, 'r-', linewidth=2.0, label='Z 3σ (m)')
    ax3.fill_between(times, 0, pos_z_sigma, alpha=0.3, color='r')
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Position Z 3σ (m)', fontsize=12)
    ax3.set_title('Z Position Uncertainty (3σ)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot Yaw 3σ
    yaw_sigma_deg = yaw_sigma * 180 / math.pi  # Convert rad to deg
    ax4.plot(times, yaw_sigma_deg, 'purple', linewidth=2.0, label='Yaw 3σ (deg)')
    ax4.fill_between(times, 0, yaw_sigma_deg, alpha=0.3, color='purple')
    
    # Add yaw trajectory for reference (right y-axis)
    ax4_yaw = None
    if yaw_deg is not None and len(yaw_deg) == len(times):
        ax4_yaw = ax4.twinx()
        ax4_yaw.plot(times, yaw_deg, 'orange', linewidth=1.0, alpha=0.5, label='Yaw (deg)')
        ax4_yaw.set_ylabel('Yaw (degrees)', fontsize=12, color='orange')
        ax4_yaw.tick_params(axis='y', labelcolor='orange')
    
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Yaw 3σ (degrees)', fontsize=12, color='purple')
    ax4.set_title('Yaw Uncertainty (3σ) with Yaw Trajectory', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    if ax4_yaw is not None:
        lines2, labels2 = ax4_yaw.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    else:
        ax4.legend(fontsize=10)
    
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.tick_params(axis='y', labelcolor='purple')
    
    # Add overall statistics
    total_time = times[-1] - times[0] if len(times) > 1 else 0
    fig.suptitle(f'Pose 3-Sigma Uncertainty Curves - {len(times)} samples, Duration: {total_time:.1f}s', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3σ uncertainty plot saved to {output_file}")
    else:
        plt.show()
    
    return fig

def main():
    # Default input directory
    default_dir = "T20260421213955"
    
    # Get input path from command line
    input_path = default_dir
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    
    # Determine if it's a directory or file
    if os.path.isdir(input_path):
        if not input_path.endswith('/'):
            input_path += '/'
        traj_file = os.path.join(input_path, "trajectory.csv")
        cpp_cov_file = os.path.join(input_path, "trajectory_covariance.txt")
        py_cov_file = os.path.join(input_path, "generated_pose_covariance.txt")
    elif os.path.isfile(input_path):
        dir_name = os.path.dirname(input_path)
        if dir_name:
            if not dir_name.endswith('/'):
                dir_name += '/'
            cpp_cov_file = os.path.join(dir_name, "trajectory_covariance.txt")
            py_cov_file = os.path.join(dir_name, "generated_pose_covariance.txt")
        else:
            cpp_cov_file = "trajectory_covariance.txt"
            py_cov_file = "generated_pose_covariance.txt"
        
        if input_path.endswith('.csv'):
            traj_file = input_path
        else:
            traj_file = None
    else:
        print(f"Error: Path not found - {input_path}")
        sys.exit(1)
    
    try:
        covariance_data = None
        traj_data = None
        source = ""
        
        # Try to load C++ generated covariance first
        if os.path.exists(cpp_cov_file):
            print(f"Loading C++ generated covariance from {cpp_cov_file}...")
            covariance_data = load_cpp_covariance(cpp_cov_file)
            if covariance_data is not None:
                source = "C++ (trajectory_covariance.txt)"
                print(f"Loaded {len(covariance_data)} covariance points from C++ output")
        
        # If C++ covariance not available, try Python generated
        if covariance_data is None and os.path.exists(py_cov_file):
            print(f"Loading Python generated covariance from {py_cov_file}...")
            covariance_data = load_python_covariance(py_cov_file)
            if covariance_data is not None:
                source = "Python (generated_pose_covariance.txt)"
                print(f"Loaded {len(covariance_data)} covariance points from Python output")
        
        # If no covariance file, compute from trajectory
        if covariance_data is None:
            if traj_file is None or not os.path.exists(traj_file):
                print(f"Error: No covariance file found and trajectory file not found at {traj_file}")
                sys.exit(1)
            
            print(f"Loading trajectory data from {traj_file}...")
            traj_data = load_trajectory(traj_file)
            
            if traj_data.shape[0] == 0:
                print(f"Error: No valid trajectory data found in {traj_file}")
                sys.exit(1)
            
            print(f"Loaded {traj_data.shape[0]} trajectory points")
            
            # Extract data
            times = traj_data[:, 0]
            positions = traj_data[:, 1:4]  # x, y, z
            
            # Convert quaternions to Euler angles
            print("Converting quaternions to Euler angles...")
            qx, qy, qz, qw = traj_data[:,4], traj_data[:,5], traj_data[:,6], traj_data[:,7]
            roll, pitch, yaw = quaternion_to_euler_batch(qx, qy, qz, qw)
            euler_angles = np.column_stack([roll, pitch, yaw])
            
            # Compute covariance from trajectory
            print("Computing pose covariance from trajectory analysis...")
            covariance_data = compute_trajectory_covariance(times, positions, euler_angles)
            source = "Computed from trajectory.csv"
        
        # Generate output filenames
        if os.path.isdir(input_path):
            output_dir = input_path.rstrip('/')
        else:
            output_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else "."
        
        cov_output_file = os.path.join(output_dir, "pose_covariance_xyz_yaw.txt")
        plot_output_file = os.path.join(output_dir, "pose_3sigma_xyz_yaw.png")
        
        # Save covariance to txt file
        save_covariance_to_txt(covariance_data, cov_output_file)
        
        # Load trajectory data for plotting reference (if not already loaded)
        if traj_data is None and traj_file is not None and os.path.exists(traj_file):
            traj_data = load_trajectory(traj_file)
        
        # Plot 3-sigma curves
        plot_3sigma_curves(covariance_data, traj_data, plot_output_file)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("COVARIANCE STATISTICS SUMMARY")
        print("="*60)
        print(f"Data source: {source}")
        print(f"\nPosition uncertainty (3σ, meters):")
        print(f"  X: mean={np.mean(covariance_data[:,1]):.6f}m, min={np.min(covariance_data[:,1]):.6f}m, max={np.max(covariance_data[:,1]):.6f}m")
        print(f"  Y: mean={np.mean(covariance_data[:,2]):.6f}m, min={np.min(covariance_data[:,2]):.6f}m, max={np.max(covariance_data[:,2]):.6f}m")
        print(f"  Z: mean={np.mean(covariance_data[:,3]):.6f}m, min={np.min(covariance_data[:,3]):.6f}m, max={np.max(covariance_data[:,3]):.6f}m")
        
        print(f"\nYaw uncertainty (3σ, degrees):")
        print(f"  Yaw: mean={np.mean(covariance_data[:,4])*180/math.pi:.6f}°, min={np.min(covariance_data[:,4])*180/math.pi:.6f}°, max={np.max(covariance_data[:,4])*180/math.pi:.6f}°")
        
        print(f"\nFiles generated:")
        print(f"  1. {cov_output_file} - Pose covariance data (x, y, z, yaw)")
        print(f"  2. {plot_output_file} - 3σ uncertainty curves plot")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("\nUsage: python3 generate_pose_covariance_and_plot.py [directory_or_file]")
        print("Examples:")
        print("  python3 generate_pose_covariance_and_plot.py T20260421213955")
        print("  python3 generate_pose_covariance_and_plot.py T20260421213955/trajectory.csv")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import matplotlib
        import numpy as np
    except ImportError as e:
        print(f"Error: Required package not installed - {e}")
        print("Please install: pip install matplotlib numpy")
        sys.exit(1)
    
    main()