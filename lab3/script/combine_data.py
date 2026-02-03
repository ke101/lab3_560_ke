import pandas as pandas
import os

 
type = {"size": [], "normal":[], "tiled": [], "cublas": []}
for f in os.listdir('./output'):
    if f.endswith('.csv'):
        f = os.path.join('./output', f)
        df = pandas.read_csv(f)
        if not type['size']:
            type['size'] = df["Size"].tolist()
        if "tiled" in f:
            type['tiled'] = df["Time"].tolist()
        elif "cublas" in f:
            type['cublas'] = df["Time"].tolist()
        else:
            type['normal'] = df["Time"].tolist() 
final_csv = pandas.DataFrame(type)
print(final_csv)

speedup = sum(type['normal']) / sum(type['tiled'])


