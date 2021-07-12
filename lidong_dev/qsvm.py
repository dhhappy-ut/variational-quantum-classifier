import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from qiskit import BasicAer
from qiskit.ml.datasets import *
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from read_data import read_cancer_data
# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging

data = read_cancer_data()
y = data[:,0]
X = data[:,[1,3]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=79)
plt.scatter(X_train[:,0],X_train[:,1])