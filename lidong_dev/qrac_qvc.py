from math import acos, pi, sqrt

import numpy as np
from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from read_data import *

class QracFeatureMap(FeatureMap):
    """Mapping data with a custom feature map."""

    def __init__(self, feature_dimension, depth=2, entangler_map=None):
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
        self._entangler_map = None
        if self._entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in
                                   range(i + 1, self._feature_dimension)]

    def construct_circuit(self, feature_string, qr, inverse=False):
        """Construct the feature map circuit.

        Args:
            feature_string (string): 3n bit string encoding the case.
            qr (QauntumRegister): the QuantumRegister object for the circuit.
            inverse (bool): whether or not to invert the circuit.

        Returns:
            QuantumCircuit: a quantum circuit transforming data x.
        """
        n_qubit = self._feature_dimension
        qc = QuantumCircuit(qr)
        theta = acos(sqrt(0.5 + sqrt(3.0) / 6.0))
        rotationParams = {"000": (2 * theta, pi / 4, -pi / 4), "010": (2 * theta, 3 * pi / 4, -3 * pi / 4),
                          "100": (pi - 2 * theta, pi / 4, -pi / 4), "110": (pi - 2 * theta, 3 * pi / 4, -3 * pi / 4),
                          "001": (2 * theta, -pi / 4, pi / 4), "011": (2 * theta, -3 * pi / 4, 3 * pi / 4),
                          "101": (pi - 2 * theta, -pi / 4, pi / 4), "111": (pi - 2 * theta, -3 * pi / 4, 3 * pi / 4)}
        bits_list = [feature_string[i:i + 3] for i in range(0, 3*n_qubit, 3)]
        for i, bit in enumerate(bits_list):
            qc.u(*rotationParams[bit], i)
            qc.barrier()
        return qc


'''
Dummy input
training_input = {'A':['000','001','010','011','100'],'B':['101','110','111']}
test_input = {'A':['000','001','010','011','100'],'B':['101','110','111']}
'''

training_input, test_input, pre_input = data2feature(read_cancer_data())
random_seed = 111

backend = BasicAer.get_backend('qasm_simulator')

optimizer = SPSA(max_trials=200, c0=4.0, skip_calibration=True)
optimizer.set_options(save_steps=1)
feature_map = QracFeatureMap(feature_dimension=3, depth=1)
var_form = TwoLocal(3, ['ry','rz'], 'cz', reps=4)
vqc = VQC(optimizer, feature_map, var_form, training_input, test_input)
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=random_seed, seed_transpiler=random_seed)
result = vqc.run(quantum_instance)
pre_result_A = vqc.predict(pre_input['A'], quantum_instance)
pre_result_B = vqc.predict(pre_input['B'], quantum_instance)
print('Predict success ratio for negative cases')
print(list(pre_result_A[1]).count(0)/len(pre_result_A[1]))
print('Predict success ratio for postive cases')
print(list(pre_result_B[1]).count(1)/len(pre_result_B[1]))
