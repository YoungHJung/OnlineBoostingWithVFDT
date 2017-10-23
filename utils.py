import os
import csv
import numpy as np
import copy
from hoeffdingtree import *    


def get_num_instances(filepath):
    ''' Count the number of data
    Args:
        filepath (string): File path
    Returns:
        (int): The number of rows
    '''
    cnt = -1 # Considering header
    with open(filepath, 'rb') as f:
        for line in f:
            cnt += 1
    return cnt

def get_rows(filepath):
    ''' Read the file and returns list of lists
    Args:
        filepath (string): File path
    Returns:
        (list): List of row lists
    '''
    rows = []
    with open(filepath, 'rb') as f:
        r = csv.reader(f)
        r.next()
        for row in r:
            rows.append(row)
    return rows

def shuffle(rows, seed=1):
    ''' Shuffle the data 
    Args:
        seed (int): Random seed
    Returns:
        (list) Shuffled list
    '''
    np.random.seed(seed)
    ret = np.random.permutation(rows)
    return list(ret)

def convert_to_instance(dataset, X, Y=None):
    ''' Convert a list to Instance
    Args:
        dataset (Dataset): The dataset that the instance belongs to
        X (list): Data
        Y (str): Label
    Returns:
        (Instance): Converted instance
    '''
    inst_values = list(copy.deepcopy(X))
    inst_values.insert(dataset.class_index(), Y)

    for i in range(len(inst_values)):
        if i == dataset.class_index() and Y is None:
            continue
        if dataset.attribute(index=i).type() == 'Nominal':
            inst_values[i] = int(dataset.attribute(index=i)
                .index_of_value(str(inst_values[i])))
        else:
            inst_values[i] = float(inst_values[i])

    ret = Instance(att_values=inst_values)
    ret.set_dataset(dataset)
    return ret

def train_vfdt(vfdt, dataset, train_rows):
    ''' Train a tree 
    Args:
        vfdt (Hoeffdingtree): Tree
        dataset (Dataset): The dataset that training set belongs to
        train_rows (list): List of data
    Return:
        (Hoeffdingtree): Trained tree
    '''
    for row in train_rows:
        X = row[1:]
        Y = row[0]
        inst = convert_to_instance(dataset, X, Y)
        vfdt.update_classifier(inst)
    return vfdt

def test_vfdt(vfdt, dataset, test_rows):
    ''' Test the trained tree
    Args:
        vfdt (Hoeffdingtree): Tree
        dataset (Dataset): The dataset that test set belongs to
        test_rows (list): List of data
    Return:
        (float): Error in percentage
    '''
    cnt = 0
    for row in test_rows:
        X = row[1:]
        Y = row[0]
        unlabeled_inst = convert_to_instance(dataset, X)
        labeled_inst = convert_to_instance(dataset, X, Y)
        pdf = vfdt.distribution_for_instance(unlabeled_inst)
        pred_index = np.argmax(pdf)
        pred = dataset.class_attribute().value(pred_index)
        if pred != Y:
            cnt += 1
        vfdt.update_classifier(labeled_inst)
    return float(cnt) / len(test_rows) * 100