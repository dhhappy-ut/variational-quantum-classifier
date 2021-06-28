import numpy as np


def read_cancer_data():
    """
    Read the breast-cancer.data file, returns np.ndarray(286,10) object contains
    0 : recurrence-events
    1 : age
    2 : menopause
    3 : tumor-size
    4 : inv-nodes
    5 : node caps
    6 : deg-malig
    7 : breast
    8 : breast-quad
    9 : irradiat
    :return: np.ndarray(286,10) object
    """
    with open('breast-cancer.data', mode='r') as cancer_file:
        pattern_list = []
        for line in cancer_file:
            pattern = np.zeros(10)
            items = line.split(',')
            # item 0
            if items[0] == 'no-recurrence-events':
                pattern[0] = 0
            else:
                pattern[0] = 1
            # item 1 age
            pattern[1] = int(items[1].split('-')[0])
            # item 2 menopause 0:it40 1:ge40 2:premeno
            if items[2] == 'it40':
                pattern[2] = 0
            elif items[2] == 'ge40':
                pattern[2] = 1
            elif items[2] == 'premeno':
                pattern[2] = 2
            # item 3 tumor size
            pattern[3] = int(items[3].split('-')[0])
            # item 4 inv nodes
            pattern[4] = int(items[4].split('-')[0])
            # item 5 node caps
            if items[5] == 'yes':
                pattern[5] = 1
            else:
                pattern[5] = 0
            # item 6 deg-malig
            pattern[6] = int(items[6])
            # item 7 breast left:0 right:1
            if items[7] == 'left':
                pattern[7] = 0
            else:
                pattern[7] = 1
            # item 8 breast quad left-up:0 left-low:1 right-up:2 right-low:3 central:4
            if items[8] == 'left_up':
                pattern[8] = 0
            elif items[8] == 'left_low':
                pattern[8] = 1
            elif items[8] == 'right_up':
                pattern[8] = 2
            elif items[8] == 'right_low':
                pattern[8] = 3
            elif items[8] == 'central':
                pattern[8] = 4
            # item 9 irradiat
            if items[9][:2] == 'no':
                pattern[9] = 0
            else:
                pattern[9] = 1
            pattern_list.append(pattern)
        return np.array(pattern_list,dtype=np.int)

