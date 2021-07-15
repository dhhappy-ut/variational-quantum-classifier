import numpy as np

np.random.seed(72)

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


def data2feature(data):
    """
    Idea: 9-bits representation for each case
    1:age: <30, 40, 50, >60
    2:tumor-size <15, 20, 25, >30
    3:deg-malig: 1:00 2:01 3:10
    4:breast-quad: lu:000 ll:001 ru:010 rl:011 c:100

    # Team A:fitted on tumor-size
    tumor-size: 4
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
    deg-malig: 2
        1 = 00
        2 = 01
        3 = 10
    nodecaps: 1
        0 = 0
        1 = 1
    irradiat: 1
        0 = 0
        1 = 1
    inv-nodes: 1
        0 or 1 = 0
        >3 = 1

    3 : tumor-size
    4 : inv-nodes
    5 : node caps
    6 : deg-malig
    7 : breast
    8 : breast-quad
    9 : irradiat

    
    """
    dict = {}
    A_list = []
    B_list = []
    for case in data:
        if case[3]==50:
            b1='1010'
        elif case[3]==45:
            b1='1001'
        elif case[3]==40:
            b1='1000'
        elif case[3]==35:
            b1='0111'
        elif case[3]==30:
            b1='0110'
        elif case[3]==25:
            b1='0101'
        elif case[3]==20:
            b1='0100'
        elif case[3]==15:
            b1='0011'
        elif case[3]==10:
            b1='0010'
        elif case[3]==5:
            b1='0001'
        else:
            b1='0000'

        if case[4]<=1:
            b3='0'
        else:
            b3='1'

        if case[5]==1:
            b5='0'
        else:
            b5='1'

        if case[6]==1:
            b7='000'
        elif case[6]==2:
            b7='001'
        elif case[6]==3:
            b7='010'

        if case[9]==0:
            b9='0'
        elif case[9]==1:
            b9='1'

        if case[0]==0:
            A_list.append(b1+b3+b5+b7+b9)
        else:
            B_list.append(b1+b3+b5+b7+b9)
    B_list = B_list + B_list
    # split data set train 8 : test 2
    training_d = {'A':A_list[:int(0.8*len(A_list))], 'B':B_list[:int(0.8*len(B_list))]}
    test_d = {'A':A_list[int(0.8*len(A_list))+1:int(len(A_list))], 'B':B_list[int(0.8*len(B_list))+1:int(len(B_list))]}
    
    # cross validation set
    random_arrayA = np.random.choice(len(A_list), len(A_list), replace=False)
    random_arrayB = np.random.choice(len(B_list), len(B_list), replace=False)
    
    vald_A_list = []
    vald_B_list = []
    for i in range(len(A_list)):
        vald_A_list.append(A_list[random_arrayA[i]])
    for i in range(len(B_list)):
        vald_B_list.append(B_list[random_arrayB[i]])

    A_validation_set = np.array_split(vald_A_list, 10)
    B_validation_set = np.array_split(vald_B_list, 10)

    pred_d = {'A':A_validation_set, 'B':B_validation_set}
    return training_d, test_d, pred_d


def str2vec(string):
    """
    transfer binary string to number:
    '00110101' -> [0,0,1,1,0,1,0,1]
    :return:
    """
    data = []
    for case in string:
        item = []
        for bin in case:
            if bin == '0':
                item.append(0)
            else:
                item.append(1)
        data.append(item)
    return np.array(data)

