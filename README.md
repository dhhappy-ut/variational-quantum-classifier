# Trainable-Embedding-QML
Implementation of Quantum Random Access Coding (QRAC) and Variational Quantum Classifier (VQC)

# Preliminaries
## Dataset
[Breast Cancer Dataset by UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer)

## (3, 1) QRAC
For implementing, we make $n$ copies of (3,1) QRAC to be a $(3n,n)$ QRAC, so that we can train a model which can accept $3n$ binary features. 

Due to our time limit, we set $n=3$.

# Set features to predict
`read_data.py` prepares data to predict on VQC.

First, we made features of breast cancer dataset as numerical form.

```
- recurrence events
    - 0: no recurrence event
    - 1: recurrence event
- age
    - replace to minimum of each interval (e.g. 20-29 to 20)
- menopuase
    - 0: it40
    - 1: ge40
    - 2: premeno
- tumor size
    - replace to minimum of each interval (e.g. 5-9 to 5)
- inv-nodes
    - replace to minimum of each interval (e.g. 3-5 to 3)
- nodecaps
    - 0: no
    - 1: yes
- breast
    - 0: left
    - 1: right
- breast-quad
    - 0: left-up
    - 1: left-low
    - 2: right-up
    - 3: right-low
    - 4: central
- irradiat
    - 0: no
    - 1: yes
```

Those features are stored in the list with indices below:
```0 : recurrence-events
1 : age
2 : menopause
3 : tumor-size
4 : inv-nodes
5 : node caps
6 : deg-malig
7 : breast
8 : breast-quad
9 : irradiat
```

---

And we choose features to use for prediction.
As our preliminary, we can only use 9 bits in total.

Our choices for the experiment are as following.:

```
- 4 bits for tumor-size
    0-4 = 0000
    5-9 = 0001
    10-14 = 0010
    15-19 = 0011
    20-24 = 0100
    25-29 = 0101
    30-34 = 0110
    35-39 = 0111
    40-44 = 1000
    45-49 = 1001
    50-54 = 1010

- 2 bits for deg-malig
    1 = 00
    2 = 01
    3 = 10

- 1 bit for nodecaps
    0 = 0
    1 = 1

- 1 bit for irradiat
    0 = 0
    1 = 1

- 1 bit for inv-nodes
    0 or 1 = 0
    >3 = 1
```
# Double Positive Cases
We duplicate positive cases because target classes are unbalanced(200 negatives, 83 positives -> 200 negatives, 166 positives)

# Cross Validation Set
We make `k` folds from randomly shuffled train set as cross validation sets.

Our default setting is `k=10`.

# Run on simulator
Run

```python qrac-qvc-simulator.py```

# Run on real backend
Run

```python qrac-qvc-real-backends.py```
