import subprocess
import matplotlib.pyplot as plt
#import time
import sys
import pandas as pd

input = sys.argv[1]
test_size = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

results = []
for s in test_size:
    try:
        result = subprocess.run([f'./{input}', str(s)],
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
    df = pd.DataFrame({'Size': sizes_list, 'Time': times_list})
    df.to_csv(f'output/{input}.csv', index=False)

else:
    print("No results to save!")