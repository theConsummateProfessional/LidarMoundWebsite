import pylas
import os
import pandas as pd

directory = 'C:\\Users\\ethan\\Development\\LidarMoundWebsite\\lazs\\'
for filename in os.listdir(directory):
    if filename.endswith('.laz'):
        print(directory + filename)
        with pylas.open(directory + filename) as fh:
            print(fh)
            las = fh.read()
            print(las)
            print('Points from Header:', fh.header.point_count)
            print(las.x)
            print(las.y)
            print(las.z)
            new_df = pd.DataFrame()
            new_df['x'] = las.x
            new_df['y'] = las.y
            new_df['z'] = las.z
            new_df.to_csv('C:\\Users\\ethan\\Development\\LidarMoundWebsite\\CSVs\\' + filename.replace('.laz', '.csv'))