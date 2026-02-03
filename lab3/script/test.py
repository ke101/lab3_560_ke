import subprocess
import matplotlib.pyplot as plt
#import time
import sys

test_size = [128, 256, 512, 1024, 2048]

results = []
for s in test_size:
    try:
        result = subprocess.run(['./matrix_cpu', str(s)],
                              capture_output=True, 
                              text=True, 
                              timeout=300)
               # Extract time from output
        for line in result.stdout.split('\n'): 
            if 'execution time' in line.lower():
                time_str = line.split(':')[-1].strip().split()[0]
                results.append((s, float(time_str)))
    except:
        print("Test size {} timed out or failed.".format(s), file=sys.stderr)
if results:
    sizes_list = [r[0] for r in results]
    times_list = [r[1] for r in results]
    plt.figure(figsize=(10, 6))
    plt.plot(sizes_list, times_list, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size (N x N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Matrix Multiplication Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
 
    plt.show()
else:
    print("no results")


