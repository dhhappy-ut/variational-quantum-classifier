from math import acos, pi, sqrt

import numpy as np
from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
IBMQ.load_account()
from read_data import *

class QracFeatureMap(FeatureMap):
    """Mapping data with a custom feature map."""

    def __init__(self, feature_dimension, depth=2):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
        """
        self._support_parameterized_circuit = False
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth

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

class TrainingMonitor:
    def __init__(self, iters, logging = True):
        self.batch_num = []
        self.params = []
        self.loss_hist = []
        self.index = []
        self.it = iters
        self.is_logging = logging
    def callback_monitor(self,a,b,c,d):
        if a%self.it==0:
            self.batch_num.append(a)
            self.params.append(b)
            self.loss_hist.append(c)
            self.index.append(d)
            if self.is_logging:
                print('Loss: %.3f'%c)
'''
Dummy input
training_input = {'A':['000','001','010','011','100'],'B':['101','110','111']}
test_input = {'A':['000','001','010','011','100'],'B':['101','110','111']}
'''

if __name__ == "__main__":
    training_input, test_input, pre_input = data2feature(read_cancer_data())
    random_seed = 333


    provider = IBMQ.get_provider(hub='ibm-q-utokyo', group='internal', project='qc2021s')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                    not x.configuration().simulator and x.status().operational==True))
    print("least busy backend: ", backend)

    optimizer = SPSA(max_trials=100, c0=4.0, skip_calibration=True)
    optimizer.set_options(save_steps=1)
    feature_map = QracFeatureMap(feature_dimension=3, depth=1)
    var_form = TwoLocal(3, ['ry','rz'], 'cz', reps=2)
    monitor = TrainingMonitor(5, logging=True)
    vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, callback=monitor.callback_monitor)
    quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=random_seed, seed_transpiler=random_seed)
    result = vqc.run(quantum_instance)

    result_set_A = []
    result_set_B = []
    for i in range(10):
        pre_result_A = vqc.predict(pre_input['A'][i], quantum_instance)
        pre_result_B = vqc.predict(pre_input['B'][i], quantum_instance)
        success_ratio_A = list(pre_result_A[1]).count(0)/len(pre_result_A[1])
        success_ratio_B = list(pre_result_B[1]).count(1)/len(pre_result_B[1])
        print(f'Predict success ratio for negative cases of {i+1}th valid set: {success_ratio_A:.3f}')
        print(f'Predict success ratio for postive cases of {i+1}th valid set: {success_ratio_B:.3f}\n')
        result_set_A.append(success_ratio_A)
        result_set_B.append(success_ratio_B)
        
    average_ratio_A = sum(result_set_A) / len(result_set_A)
    average_ratio_B = sum(result_set_B) / len(result_set_B)
    print(f"success negative cases: {average_ratio_A:.3f}\n success positive cases: {average_ratio_B:.3f}")

    # Calculate f1 score
    p = list(pre_result_B[1]).count(1)/(list(pre_result_B[1]).count(1)+list(pre_result_A[1]).count(1))
    r = list(pre_result_B[1]).count(1)/len(pre_result_B[1])
    f1 = 2*p*r/(p+r)
    
    overall = (list(pre_result_A[1]).count(0)+list(pre_result_B[1]).count(1))/(len(pre_result_A[1])+len(pre_result_B[1]))
    print(f'Overall success ratio: {overall:.3f}')
    print(f'F1 value: {f1:.3f}')