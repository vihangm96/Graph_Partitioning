import numpy as np
from numpy import linalg
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.metrics import accuracy_score

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

def classification_train(embeddings, label_train_filename):
    
    neigh_classifier = KNeighborsClassifier(n_neighbors=5)
    
    y = []
    X = []
    
    file = open(label_train_filename, "r")
    for line in file:
        data = [int(x) for x in line.split()]
        X.append(embeddings[data[0]])
        y.append(data[1])
    
    neigh_classifier.fit(X, y)
    
    return neigh_classifier

def classification_predict(classifier, embeddings, label_test_filename):
    
    X = []
    X_nodes = []
    
    file = open(label_test_filename, "r")
    for line in file:
        node = [int(x) for x in line.split()][0]
        X.append(embeddings[node])
        X_nodes.append(node)
        
    class_labels = classifier.predict(X)
    
    return X_nodes, class_labels

if __name__ == "__main__":

    edge_filename = sys.argv[1]
    label_train_filename = sys.argv[2]
    label_test_filename = sys.argv[3]
    output_file = sys.argv[4]

    #edge_filename = "data/email-Eu-core.txt"
    #output_file = "task4_output.csv"
    #label_train_filename = "data/labels_train.csv"
    #label_test_filename = "data/labels_test.csv"

    L = load_data(edge_filename)
    eigenpairs = get_eigenpairs(L)
    embeddings = eigenpairs[1][:,:105]
    classifier = classification_train(embeddings, label_train_filename)
    test_nodes, node_labels = classification_predict(classifier, embeddings, label_test_filename)
    
    solution_string = ''        
    for i in range(len(node_labels)):
        solution_string += str(test_nodes[i])+' '+str(node_labels[i])+'\n'
    out_file = open(output_file, "w")  
    out_file.write(solution_string)
    out_file.close()  