import numpy as np
from numpy import linalg
from sklearn.cluster import KMeans
import sys
from sklearn.metrics.cluster import adjusted_rand_score

def load_data(input_file):
    edges = []
    file = open(input_file, "r")
    for line in file:
        nodes = [int(x) for x in line.split()]
        edges.append([nodes[0],nodes[1]])
    
    mat_size = max(max([edge[0] for edge in edges]), max([edge[1] for edge in edges])) + 1

    L = np.zeros((mat_size, mat_size)) 
        
    for edge in edges:
        if edge[0] != edge[1]:
            L[edge[0]][edge[1]] = -1
            L[edge[1]][edge[0]] = -1
    
    for node in range(mat_size):
        L[node][node] = -1 * sum(L[node])
                
    return L

def get_eigenpairs(A):
    eigenValues, eigenVectors = linalg.eig(A)
    eigenValues = np.real(eigenValues)
    eigenVectors = np.real(eigenVectors)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def clustering(L, k):
    eigenpairs = get_eigenpairs(L)
    embeddings = eigenpairs[1][:,19:23]
    clusters = KMeans(n_clusters=k, random_state=0).fit(embeddings)
    node_labels = dict()
    
    return clusters.labels_

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    k = int(sys.argv[3])

    #input_file = "data/email-Eu-core.txt"
    #output_file = "task3_output.txt"
    #k = 42

    L = load_data(input_file)
    node_labels = clustering(L,k)

    solution_string = ''

    for node in range(len(L)):
        solution_string += str(node)+' '+str(node_labels[node])+'\n'

    out_file = open(output_file, "w")  
    out_file.write(solution_string)
    out_file.close()  