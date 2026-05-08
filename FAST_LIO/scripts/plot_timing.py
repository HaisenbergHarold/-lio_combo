#!/usr/bin/env python3
"""
Plot match_time and total_time per frame from timing_log.txt
Format per line: match_time: xxx, total_time: xxx
Parsed via regex.
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        data_path = "../FF-LINS-main/src/FAST_LIO/Log/timing_log.txt"
    else:
        data_path = sys.argv[1]

    match_times = []
    total_times = []

    pattern = re.compile(r"match_time:\s*([\d.eE+-]+),\s*total_time:\s*([\d.eE+-]+)")

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.search(line)
            if m:
                match_times.append(float(m.group(1)))
                total_times.append(float(m.group(2)))

    if not match_times:
        print(f"No data found in {data_path}")
        return

    match_times = np.array(match_times) * 1000.0   # s -> ms
    total_times = np.array(total_times) * 1000.0
    frames = np.arange(len(match_times))

    print(f"Read {len(frames)} frames from {data_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(frames, match_times, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('match_time (ms)', color='b')
    ax1.set_title('Per-Frame Timing Analysis')
    ax1.grid(True, alpha=0.3)
    mean_m = np.mean(match_times)
    ax1.axhline(y=mean_m, color='r', linestyle='--', linewidth=1,
                label=f'Average: {mean_m:.3f} ms')
    ax1.legend(loc='upper right')

    ax2.plot(frames, total_times, 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('total_time (ms)', color='r')
    ax2.grid(True, alpha=0.3)
    mean_t = np.mean(total_times)
    ax2.axhline(y=mean_t, color='b', linestyle='--', linewidth=1,
                label=f'Average: {mean_t:.3f} ms')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    out_path = data_path.rsplit('.', 1)[0] + '_timing.png'
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to: {out_path}")
    print(f"match_time - min: {np.min(match_times):.3f} ms, max: {np.max(match_times):.3f} ms, avg: {mean_m:.3f} ms")
    print(f"total_time - min: {np.min(total_times):.3f} ms, max: {np.max(total_times):.3f} ms, avg: {mean_t:.3f} ms")

    plt.show()

if __name__ == '__main__':
    main()