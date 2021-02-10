import numpy as np
from numpy import linalg
import json
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys

def load_data(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            coordinates = json.loads(line)
    
    neighbor_count = 8
    
    neigh = NearestNeighbors(n_neighbors=neighbor_count)
    neigh.fit(coordinates)

    A = np.zeros((len(coordinates),len(coordinates)))
    
    for i in range(len(coordinates)):
        for nearest_neighbor in neigh.kneighbors([coordinates[i]], return_distance=False)[0]:
            if i != nearest_neighbor:
                A[nearest_neighbor][i] = 1
                A[i][nearest_neighbor] = 1
                
    D = np.zeros((len(coordinates),len(coordinates)))
    
    for i in range(len(coordinates)):
        D[i][i] = sum(A[i])
    
    L = D-A
    
    return coordinates, L

def get_eigenpairs(A):
    eigenValues, eigenVectors = linalg.eig(A)
    eigenValues = np.real(eigenValues)
    eigenVectors = np.real(eigenVectors)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues,eigenVectors.T)

if __name__ == "__main__":

    edge_filename = sys.argv[1]
    output_filename = sys.argv[2]

    #edge_filename = "data/non-spherical.json"
    #output_filename = "output5.png"

    coordinates, L = load_data(edge_filename)

    eigenpairs = get_eigenpairs(L)
    lambda2 = eigenpairs[1][1]

    mean = np.mean(lambda2)


    labelP = []
    labelN = []
    for i in range(len(coordinates)):    
        if lambda2[i] >= mean:
            labelP.append(coordinates[i])
        else:
            labelN.append(coordinates[i])

    plt.scatter([c[0] for c in labelN], [c[1] for c in labelN], color='purple')
    plt.scatter([c[0] for c in labelP], [c[1] for c in labelP], color='yellow')
    plt.savefig(output_filename)