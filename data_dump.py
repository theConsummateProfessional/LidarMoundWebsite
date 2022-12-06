from msilib.schema import Directory
import pandas as pd
import numpy
import time
import matplotlib.pyplot as plt
import math
import os
import sys

def CalcNormalPlane(p1, p2):
    return [p1[0]*-p2[0], p1[1]*-p2[1], p1[2]*-p2[2]]

def CalcUnitVector(p1, p2):
    subtracted = numpy.subtract(p2, p1)
    return subtracted / numpy.linalg.norm(subtracted)

def CalcW(p1, p2):
    u = CalcNormalPlane(p1, p2)
    v = numpy.cross(u, CalcUnitVector(p2, p1))
    return numpy.cross(u, v)

def CalcTheta(p1, p2):
    w = CalcW(p1, p2)
    n2 = CalcNormalPlane(p2, p1)
    u = CalcNormalPlane(p1, p2)
    return math.atan(numpy.dot(w, n2) / numpy.dot(u, n2))

def main(argv1, argv2, argv3):
    for filename in os.listdir(argv1):
        f = os.path.join(argv1, filename)
        if(os.path.isfile(f) and ".csv" in f):
            name_of_mound = filename.replace(".csv", "")
            df = pd.read_csv(f)
            
            minX = min(df['x'])
            minY = min(df['y'])
            minZ = min(df['z'])
            df['x'] = df['x'] - minX
            df['y'] = df['y'] - minY
            df['z'] = df['z'] - minZ

            centroidX = df['x'].sum() / len(df.index)
            centroidY = df['y'].sum() / len(df.index)
            centroidZ = df['z'].sum() / len(df.index)
            starting_point = (centroidX, centroidY, centroidZ)

            maxim = 0
            for index, row in df.iterrows():
                radius = math.sqrt(((starting_point[0] - row['x']) ** 2) + ((starting_point[1] - row['y']) ** 2) + ((starting_point[2] - row['z']) ** 2))
                if(radius > maxim):
                    maxim = radius

            bins = []
            alphas = []
            phis = []
            thetas = []
        
            for index, row in df.iterrows():
                alphas.append(numpy.dot(CalcNormalPlane(starting_point, (row['x'], row['y'], row['z'])), CalcNormalPlane((row['x'], row['y'], row['z']), starting_point)))
                phis.append(numpy.dot(CalcNormalPlane(starting_point, (row['x'], row['y'], row['z'])), CalcUnitVector((row['x'], row['y'], row['z']), starting_point)))
                thetas.append(CalcTheta(starting_point, (row['x'], row['y'], row['z'])))
                
            alpha_range = numpy.ptp(alphas)
            phi_range = numpy.ptp(phis)
            theta_range = numpy.ptp(thetas)

            alpha_min = min(alphas)
            phi_min = min(phis)
            theta_min = min(thetas)
            alphas_normal = []
            phis_normal = []
            thetas_normal = []

            for x in range(len(alphas)):
                alpha_normal = (alphas[x] - alpha_min) / alpha_range
                phi_normal = (phis[x] - phi_min) / phi_range
                theta_normal = (thetas[x] - theta_min) / theta_range
                bins.append([alpha_normal, phi_normal, theta_normal])
                alphas_normal.append(alpha_normal)
                phis_normal.append(phi_normal)
                thetas_normal.append(theta_normal)
            
            n_bins = len(bins)
            print(argv2 + name_of_mound)
            plt.hist(alphas_normal, n_bins, density=True, histtype='bar')
            plt.savefig(argv2 + name_of_mound + '-alpha')
            plt.close()

            plt.hist(thetas_normal, n_bins, density=True)
            plt.savefig(argv2 + name_of_mound + '-theta')
            plt.close()

            plt.hist(phis_normal, n_bins, density=True)
            plt.savefig(argv2 + name_of_mound + '-phi')
            plt.close()

            new_df = pd.DataFrame()
            new_df['alphas'] = alphas_normal
            new_df['theta'] = thetas_normal
            new_df['phi'] = phis_normal
            new_df.to_csv(argv3 + name_of_mound + '.csv')


#path to directory with csv files as the first
#path to output directory as the second
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

