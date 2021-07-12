#Libraries~

import numpy as np
from math import sqrt, cos, acos, pi
import pandas as pd

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, ClassicalRegister, QuantumRegister, execute, BasicAer
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *

from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms import VQC
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

from sklearn.model_selection import train_test_split

import random

from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import BlueprintCircuit


#Generate the random bitstring with length 3 and the target
def rand_bits(n_data):
    x = []
    y = []
    for i in range(n_data):
        b = str(random.randint(0, 1))+str(random.randint(0, 1))+str(random.randint(0, 1))
        x.append(b)
        y.append(str(random.randint(0, 1)))
    return x,y
# As Lidong mentioned these parts is not represented the classification because the target are also random
dummy_data = rand_bits(50)


#Splitting
X_train, X_test, y_train, y_test = train_test_split(dummy_data[0], dummy_data[1], test_size=0.2)

#Change to dictionary
training_input = { 'A': [X_train[i] for i in range(len(X_train)) if y_train[i] == '0'], 'B': [X_train[i] for i in range(len(X_train)) if y_train[i] == '1']}
test_input = { 'A': [X_test[i] for i in range(len(X_test)) if y_test[i] == '0'], 'B': [X_test[i] for i in range(len(X_test)) if y_test[i] == '1']}

#Pick 10 for datapoints
extra_test_data = test_input = { 'A': [X_test[i] for i in range(len(X_test)) if y_test[i] == '0'][:10], 'B': [X_test[i] for i in range(len(X_test)) if y_test[i] == '1'][:10]}
datapoints, class_to_label = split_dataset_to_data_and_labels(extra_test_data)

#Parameters for VQC
optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
var_form = TwoLocal(1, ['ry', 'rz'], 'cz', reps=3) #Choose 1 because only 1 qubit

class CustomFeatureMap(FeatureMap):
    """Mapping data with a custom feature map."""
    
    def __init__(self, feature_dimension, depth):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.        
        """
        self._support_parameterized_circuit = False
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth
            
    def construct_circuit(self, x, qr):
        """Construct the feature map circuit.
        
        Args:
            x : 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit.
            
        Returns:
            QuantumCircuit: a quantum circuit transforming data x.
        """
        qc = QuantumCircuit(qr)
        #compute the value of theta
        theta = acos(sqrt(0.5 + sqrt(3.0)/6.0))

        #to record the u3 parameters for encoding 000, 010, 100, 110, 001, 011, 101, 111
        rotationParams = {"000":(2*theta, pi/4, -pi/4), "010":(2*theta, 3*pi/4, -3*pi/4), 
                          "100":(pi-2*theta, pi/4, -pi/4), "110":(pi-2*theta, 3*pi/4, -3*pi/4), 
                          "001":(2*theta, -pi/4, pi/4), "011":(2*theta, -3*pi/4, 3*pi/4), 
                          "101":(pi-2*theta, -pi/4, pi/4), "111":(pi-2*theta, -3*pi/4, 3*pi/4)}
        
        qc.u(*rotationParams[x], qr[0])
        return qc

feature_map = CustomFeatureMap(feature_dimension=1, depth=1) #1 qubit and only need 1 repetition circuit

#Execute
vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, datapoints[0])
seed =1024
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

result = vqc.run(quantum_instance)

#For results
print(f'Testing success ratio: {result["testing_accuracy"]}')
print()
print('Prediction from datapoints set:')
print(f'  ground truth: {map_label_to_class_name(datapoints[1], vqc.label_to_class)}')
print(f'  prediction:   {result["predicted_classes"]}')
predicted_labels = result["predicted_labels"]
print(f'  success rate: {100*np.count_nonzero(predicted_labels == datapoints[1])/len(predicted_labels)}%')
