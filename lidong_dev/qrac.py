# useful math functions
from math import pi, acos, sqrt

# importing QISKit
from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


def qrac_feature_map(feature_string, n_qubits):
    theta = acos(sqrt(0.5 + sqrt(3.0)/6.0))
    rotationParams = {"000": (2 * theta, pi / 4, -pi / 4), "010": (2 * theta, 3 * pi / 4, -3 * pi / 4),
                      "100": (pi - 2 * theta, pi / 4, -pi / 4), "110": (pi - 2 * theta, 3 * pi / 4, -3 * pi / 4),
                      "001": (2 * theta, -pi / 4, pi / 4), "011": (2 * theta, -3 * pi / 4, 3 * pi / 4),
                      "101": (pi - 2 * theta, -pi / 4, pi / 4), "111": (pi - 2 * theta, -3 * pi / 4, 3 * pi / 4)}
    bits_list = [feature_string[i:i+3] for i in range(0, 3*n_qubits,3)]
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr, name='qrac_feature_map')
    for i, bit in enumerate(bits_list):
        qc.u(*rotationParams[bit], i)
        qc.barrier()
    return qc

if __name__  == '__main__':
    theta = acos(sqrt(0.5 + sqrt(3.0) / 6.0))
    backend = Aer.get_backend('qasm_simulator') # the device to run on
    shots = 1024    # the number of shots in the experiment
    #to record the u3 parameters for encoding 000, 010, 100, 110, 001, 011, 101, 111
    rotationParams = {"000":(2*theta, pi/4, -pi/4), "010":(2*theta, 3*pi/4, -3*pi/4),
                      "100":(pi-2*theta, pi/4, -pi/4), "110":(pi-2*theta, 3*pi/4, -3*pi/4),
                      "001":(2*theta, -pi/4, pi/4), "011":(2*theta, -3*pi/4, 3*pi/4),
                      "101":(pi-2*theta, -pi/4, pi/4), "111":(pi-2*theta, -3*pi/4, 3*pi/4)}
    # Creating registers
    # qubit for encoding 3 bits of information
    qr = QuantumRegister(1)
    # bit for recording the measurement of the qubit
    cr = ClassicalRegister(1)
    # dictionary for encoding circuits
    encodingCircuits = {}
    # Quantum circuits for encoding 000, ..., 111
    for bits in rotationParams.keys():
        circuitName = "Encode"+bits
        encodingCircuits[circuitName] = QuantumCircuit(qr, cr, name=circuitName)
        encodingCircuits[circuitName].u(*rotationParams[bits], qr[0])
        encodingCircuits[circuitName].barrier()

    # dictionary for decoding circuits
    decodingCircuits = {}
    # Quantum circuits for decoding the first, second and third bit
    for pos in ("First", "Second", "Third"):
        circuitName = "Decode"+pos
        decodingCircuits[circuitName] = QuantumCircuit(qr, cr, name=circuitName)
        if pos == "Second": #if pos == "First" we can directly measure
            # This gate: |+> -> |0>, |-> -> |1>
            decodingCircuits[circuitName].h(qr[0])
        elif pos == "Third":
            # This gate: |Y+> -> |0>, |Y-> -> |1> (|Y+> = (1, i)^T, |Y-> = (1,-i)^T)
            decodingCircuits[circuitName].u(pi/2, -pi/2, pi/2, qr[0])
        decodingCircuits[circuitName].measure(qr[0], cr[0])

    #combine encoding and decoding of QRACs to get a list of complete circuits
    circuitNames = []
    circuits = []
    for k1 in encodingCircuits.keys():
        for k2 in decodingCircuits.keys():
            circuitNames.append(k1+k2)
            circuits.append(encodingCircuits[k1]+decodingCircuits[k2])

    print("List of circuit names:", circuitNames) #list of circuit names