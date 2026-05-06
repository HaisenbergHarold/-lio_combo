#!/usr/bin/env python3
"""
Plot per-keyframe timing statistics for FF-LINS.
Reads timing_statistic.txt and generates plots showing:
  - Per-keyframe time costs for each function (subplots)
  - Average time costs (bar chart)

Input format (timing_statistic.txt, 10 columns):
  timestamp total merge add_node update_feature lins_processing optimization redo_ins add_cloud remaining
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def load_timing_data(filename):
    """Load timing data from timing_statistic.txt (10 columns)."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 10:
                try:
                    row = [float(x) for x in parts[:10]]
                    data.append(row)
                except ValueError:
                    continue
    if not data:
        return None
    arr = np.array(data)
    # Sort by timestamp
    sorted_idx = np.argsort(arr[:, 0])
    arr = arr[sorted_idx]
    return arr


def plot_timing_curves(timing_data, output_dir=None):
    """Plot per-keyframe timing curves with 4 subplots."""
    times = timing_data[:, 0]
    cols = {
        'total':            timing_data[:, 1],
        'merge':            timing_data[:, 2],
        'add_node':         timing_data[:, 3],
        'update_feature':   timing_data[:, 4],
        'lins_processing':  timing_data[:, 5],
        'optimization':     timing_data[:, 6],
        'redo_ins':         timing_data[:, 7],
        'add_cloud':        timing_data[:, 8],
        'remaining':        timing_data[:, 9],
    }

    elapsed = times - times[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('FF-LINS Per-Keyframe Timing Statistics', fontsize=16, fontweight='bold', y=0.98)

    colors = {
        'total':            '#1f77b4',
        'merge':            '#ff7f0e',
        'add_node':         '#2ca02c',
        'update_feature':   '#d62728',
        'lins_processing':  '#9467bd',
        'optimization':     '#8c564b',
        'redo_ins':         '#e377c2',
        'add_cloud':        '#7f7f7f',
        'remaining':        '#bcbd22',
    }

    # Subplot arrangement:
    # (0,0): Total time
    # (0,1): mergeNonKeyframes + addNewLidarFrameTimeNode
    # (1,0): updateLidarFeaturesInMap + linsLidarProcessing
    # (1,1): optimization + redoInsMechanization + addNewPointCloudToMap

    subplots = [
        (0, 0, ['total'], 'Total Time Per Keyframe (ms)'),
        (0, 1, ['merge', 'add_node'], 'Frame Merging & Node Adding (ms)'),
        (1, 0, ['update_feature', 'lins_processing'], 'Feature Processing (ms)'),
        (1, 1, ['optimization', 'redo_ins', 'add_cloud'], 'Optimization & Post-Processing (ms)'),
    ]

    for row, col, keys, title in subplots:
        ax = axes[row, col]
        for key in keys:
            ax.plot(elapsed, cols[key], color=colors[key], linewidth=1.5,
                    label=key, alpha=0.85)
            ax.fill_between(elapsed, 0, cols[key], color=colors[key], alpha=0.08)

        ax.set_xlabel('Elapsed Time (s)', fontsize=11)
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_file = os.path.join(output_dir, 'timing_curves.png') if output_dir else 'timing_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Timing curves saved to {plot_file}")
    plt.close()


def plot_average_bar(timing_data, output_dir=None):
    """Plot average time costs as a bar chart."""
    cols = {
        'merge':            timing_data[:, 2],
        'add_node':         timing_data[:, 3],
        'update_feature':   timing_data[:, 4],
        'lins_processing':  timing_data[:, 5],
        'optimization':     timing_data[:, 6],
        'redo_ins':         timing_data[:, 7],
        'add_cloud':        timing_data[:, 8],
    }
    total_ms = timing_data[:, 1]

    labels_list = ['merge\nNonKeyframes', 'addNewLidar\nFrameTimeNode',
                   'updateLidar\nFeaturesInMap', 'linsLidar\nProcessing',
                   'lins\nOptimization', 'redoIns\nMechanization', 'addNew\nPointCloud']
    values_list = [np.mean(cols[k]) for k in ['merge', 'add_node', 'update_feature',
                                               'lins_processing', 'optimization', 'redo_ins', 'add_cloud']]
    stds_list   = [np.std(cols[k]) for k in ['merge', 'add_node', 'update_feature',
                                              'lins_processing', 'optimization', 'redo_ins', 'add_cloud']]

    colors_bar = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Create two figures: one for breakdown, one for total
    # Figure 1: Breakdown
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.suptitle('Average Per-Keyframe Timing Breakdown', fontsize=16, fontweight='bold', y=0.98)

    x = np.arange(len(labels_list))
    bars = ax1.bar(x, values_list, yerr=stds_list, capsize=8, color=colors_bar,
                   edgecolor='black', linewidth=1.2, width=0.6, alpha=0.85)

    for bar, val, std in zip(bars, values_list, stds_list):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
                 f'{val:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Average Time (ms)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_list, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_file1 = os.path.join(output_dir, 'timing_average_breakdown.png') if output_dir else 'timing_average_breakdown.png'
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    print(f"Average breakdown saved to {plot_file1}")
    plt.close()

    # Figure 2: Total + component stack
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Average Total Time Per Keyframe', fontsize=16, fontweight='bold', y=0.98)

    total_mean = np.mean(total_ms)
    total_std  = np.std(total_ms)

    ax2.bar(['Total'], [total_mean], yerr=[total_std], capsize=10,
            color='#1f77b4', edgecolor='black', linewidth=1.5, width=0.4, alpha=0.85)

    ax2.text(0, total_mean + total_std + 1, f'{total_mean:.2f} ms',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_ylabel('Average Time (ms)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Print summary
    print("\n" + "=" * 60)
    print("TIMING STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total keyframes: {len(timing_data)}")
    print(f"\nAverage time per keyframe (mean ± std, ms):")
    all_labels = ['mergeNonKeyframes', 'addNewLidarFrameTimeNode',
                  'updateLidarFeaturesInMap', 'linsLidarProcessing',
                  'linsOptimizationProcessing', 'redoInsMechanization',
                  'addNewPointCloudToMap', 'Total']
    all_values = values_list + [total_mean]
    all_stds   = stds_list + [total_std]
    for label, val, std in zip(all_labels, all_values, all_stds):
        print(f"  {label:30s}: {val:8.2f} ± {std:6.2f}")

    print(f"\nMaximum time per keyframe (ms):")
    max_keys = ['total', 'merge', 'add_node', 'update_feature',
                'lins_processing', 'optimization', 'redo_ins', 'add_cloud']
    max_labels = ['Total', 'mergeNonKeyframes', 'addNewLidarFrameTimeNode',
                  'updateLidarFeaturesInMap', 'linsLidarProcessing',
                  'linsOptimizationProcessing', 'redoInsMechanization',
                  'addNewPointCloudToMap']
    for label, key in zip(max_labels, max_keys):
        if key == 'total':
            print(f"  {label:30s}: {np.max(timing_data[:, 1]):8.2f}")
        else:
            print(f"  {label:30s}: {np.max(cols[key]):8.2f}")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_file2 = os.path.join(output_dir, 'timing_average_total.png') if output_dir else 'timing_average_total.png'
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"\nAverage total saved to {plot_file2}")
    plt.close()


def plot_all_overlay(timing_data, output_dir=None):
    """Plot all components overlaid on a single figure."""
    times = timing_data[:, 0]
    elapsed = times - times[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('All Timing Components Overlay', fontsize=16, fontweight='bold', y=0.98)

    keys = ['total', 'merge', 'add_node', 'update_feature', 'lins_processing',
            'optimization', 'redo_ins', 'add_cloud', 'remaining']
    colors = {
        'total': '#1f77b4', 'merge': '#ff7f0e', 'add_node': '#2ca02c',
        'update_feature': '#d62728', 'lins_processing': '#9467bd',
        'optimization': '#8c564b', 'redo_ins': '#e377c2',
        'add_cloud': '#7f7f7f', 'remaining': '#bcbd22',
    }

    for key in keys:
        if key == 'total':
            ax.plot(elapsed, timing_data[:, 1], color=colors[key], linewidth=2.5,
                    label=key, alpha=0.9)
        else:
            idx = {'merge': 2, 'add_node': 3, 'update_feature': 4,
                   'lins_processing': 5, 'optimization': 6, 'redo_ins': 7,
                   'add_cloud': 8, 'remaining': 9}[key]
            ax.plot(elapsed, timing_data[:, idx], color=colors[key], linewidth=1.2,
                    label=key, alpha=0.7)

    ax.set_xlabel('Elapsed Time (s)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_file = os.path.join(output_dir, 'timing_all_overlay.png') if output_dir else 'timing_all_overlay.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"All overlay saved to {plot_file}")
    plt.close()


def main():
    input_path = None
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

    candidates = []
    if input_path:
        if os.path.isdir(input_path):
            f = os.path.join(input_path, 'timing_statistic.txt')
            if os.path.exists(f):
                candidates.append(f)
        elif os.path.isfile(input_path):
            if input_path.endswith('timing_statistic.txt'):
                candidates.append(input_path)
    else:
        for root, dirs, files in os.walk('.'):
            if 'timing_statistic.txt' in files:
                candidates.append(os.path.join(root, 'timing_statistic.txt'))

    if not candidates:
        print("Error: timing_statistic.txt not found.")
        print("Usage: python3 plot_timing_statistic.py [directory_or_file]")
        print("  - If no argument: searches current directory recursively")
        print("  - If directory: looks for timing_statistic.txt inside")
        print("  - If file: uses that file directly")
        sys.exit(1)

    timing_file = candidates[0]
    output_dir = os.path.dirname(timing_file)

    print(f"Loading timing data from: {timing_file}")
    timing_data = load_timing_data(timing_file)

    if timing_data is None or len(timing_data) == 0:
        print("Error: No valid timing data found.")
        sys.exit(1)

    print(f"Loaded {len(timing_data)} keyframe timing records")

    plot_timing_curves(timing_data, output_dir)
    plot_average_bar(timing_data, output_dir)
    plot_all_overlay(timing_data, output_dir)

    print(f"\nAll plots saved to directory: {output_dir}")


if __name__ == "__main__":
    try:
        import matplotlib
        import numpy as np
    except ImportError as e:
        print(f"Error: Required package not installed - {e}")
        print("Please install: pip install matplotlib numpy")
        sys.exit(1)

    main()
