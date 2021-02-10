import numpy as np
import sys
from numpy import linalg

def get_eigenpairs(A):
    eigenValues, eigenVectors = linalg.eig(A)
    
    idx = np.argsort(eigenValues)
    
    eigenValues = np.real(eigenValues)
    eigenVectors = np.real(eigenVectors)
    
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    return (eigenValues, eigenVectors.T)

def get_important_nodes(input_file, d=0.8, tol=1e-8, max_iter=100, K=20):
    
    edges = []
    
    file = open(input_file, "r")
    for line in file:
        edges.append([int(x) for x in line.split()])
    
    N = max(max([edge[0] for edge in edges]), max([edge[1] for edge in edges])) + 1

    M = np.zeros((N, N)) 
        
    for edge in edges:
        if edge[1]!=edge[0]:
            M[edge[1]][edge[0]] = 1
    
    out_degree = M.sum(axis=0)
    
    #To avoid dead ends, since out_degree is 0, replace all the values with 1/N to make it column stochastic
    A = np.divide(M, out_degree, out=np.ones_like(M)/N, where=out_degree!=0)
    
    A = A * d + (1-d)/N 
    
    eigenpairs = get_eigenpairs(A)

    i = np.where(np.isclose(eigenpairs[0],1))[0]
    
    
    pr = eigenpairs[1][i]

    most_important_nodes = [n[0] for n in sorted(enumerate(np.asarray(pr).squeeze()), reverse=True, key=lambda x: x[1])[:K]]
    
    return most_important_nodes

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    #input_file = "data/email-Eu-core.txt"
    #output_file = "task6_output.txt"

    most_important_nodes = get_important_nodes(input_file)

    solution_string = ''

    for node in most_important_nodes:
        solution_string += str(node)+'\n'

    out_file = open(output_file, "w")  
    out_file.write(solution_string)
    out_file.close()  