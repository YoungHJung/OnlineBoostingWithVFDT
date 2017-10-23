from hoeffdingtree import *    
from onlineAdaptive import AdaBoostOLM
from oneVSall import oneVSall, oneVSallBoost
import utils

def main():
	# Load data
	filename = 'balance-scale.csv'
	class_index = 0
	training_ratio = 0.8

	N = utils.get_num_instances(filename)
	train_N = int(N*training_ratio)
	rows = utils.get_rows(filename)
	rows = utils.shuffle(rows, seed=357)

	train_rows = rows[:train_N]
	test_rows = rows[train_N:]

	# Set parameters
	num_weaklearners = 10
	gamma = 0.1
	M = 100

	print 'Num weak learners:', num_weaklearners

	# Test Adaboost.OLM
	model = AdaBoostOLM(loss='logistic')
	model.initialize_dataset(filename, class_index, N)
	dataset = model.get_dataset()
	model.gen_weaklearners(num_weaklearners,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=5, max_weight=200) 

	for i, row in enumerate(train_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)

	cnt = 0

	for i, row in enumerate(test_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)
	    cnt += (pred == Y)*1

	result = round(100 * cnt / float(len(test_rows)), 2)
	print 'Adaboost.OLM:', result

	# Test OnlineMBBM
	model = AdaBoostOLM(loss='zero_one', gamma=gamma)
	model.M = M
	model.initialize_dataset(filename, class_index, N)
	model.gen_weaklearners(num_weaklearners,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=5, max_weight=200) 

	for i, row in enumerate(train_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)

	cnt = 0

	for i, row in enumerate(test_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)
	    cnt += (pred == Y)*1

	result = round(100 * cnt / float(len(test_rows)), 2)
	print 'OnlineMBBM:', result

	# Test one vs all method
	model = oneVSall()
	model.initialize_dataset(filename, class_index, dataset.num_classes(), N)
	model.initialize_binary_learners(num_weaklearners,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=5, max_weight=200) 	

	for i, row in enumerate(train_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)

	cnt = 0

	for i, row in enumerate(test_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)
	    cnt += (pred == Y)*1

	result = round(100 * cnt / float(len(test_rows)), 2)
	print 'one vs all:', result

	# Test boosting with weak learners built upon one vs all methods
	model = oneVSallBoost()
	model.initialize_dataset(filename, class_index, N)
	model.gen_weaklearners(num_weaklearners,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=5, max_weight=200) 

	for i, row in enumerate(train_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)

	cnt = 0

	for i, row in enumerate(test_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)
	    cnt += (pred == Y)*1

	result = round(100 * cnt / float(len(test_rows)), 2)
	print 'one vs all boost:', result

if __name__ == '__main__':
	main()