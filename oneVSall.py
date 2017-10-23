import numpy as np
import copy
from onlineAdaptive import * 

class oneVSall:
    '''
    An implimentation of one vs all multiclass adaption of binary online
    classification. 
    '''

    def __init__(self):

        self.binary_learners = None
        self.dataset = None
        self.class_index = None
        self.num_classes = None
        self.class_mapping = None
        self.num_data = 0
        self.num_errors = 0
        self.weight_consts = None

        self.X = None
        self.Yhat_index = None
        self.Y_index = None
        self.Yhat = None
        self.Y = None
        
    ########################################################################

    def predict(self, X):
        '''Runs the entire prediction procedure

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset

        Returns:
            Yhat (string): The final class prediction
        '''

        self.X = X

        votes = np.zeros(self.num_classes)
        for i in xrange(self.num_classes):

            self.binary_learners[i].predict(X)
            votes[i] = self.binary_learners[i].pred_conf[0]
        
        tmp = [x for x in range(self.num_classes) if votes[x] == max(votes)]
        self.Yhat_index = np.random.choice(tmp)
        self.Yhat = self.class_mapping[self.Yhat_index]
        
        return self.Yhat

    def update(self, Y, X=None):
        '''Runs the entire updating procedure, updating interal 
        tracking of wl_weights and expert_weights
        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset. If not given
                      the last X used for prediction will be used.
            Y (string): The true class
        '''

        if X is None:
            X = self.X

        self.X = X
        self.Y = Y
        self.Y_index = np.nonzero(self.class_mapping == self.Y)
        self.num_data +=1
        self.num_errors = self.num_errors + (self.Y_index != self.Yhat_index)

        for i in xrange(self.num_classes):
            tempY = copy.deepcopy(self.Y)
            if tempY == self.class_mapping[i]:
                tempY = 'myClass'
            else:
                tempY = 'notMyClass'

            self.binary_learners[i].update(tempY, X)

            

    def initialize_dataset(self, filename, class_index, 
                                num_classes, probe_instances=10000):
        """ CODE HERE TAKEN FROM main.py OF HOEFFDINGTREE 
        Open and initialize a dataset in CSV format.
        The CSV file needs to have a header row, from where the attribute 
        names will be read, and a set of instances containing at least one 
        example of each value of all nominal attributes.

        Args:
            filename (str): The name of the dataset file (including filepath).
            class_index (int): The index of the attribute to be set as class.
            num_classes (int): Total number of classes
            probe_instances (int): The number of instances to be used to 
                initialize the nominal attributes. (default 100)

        Returns:
            It does not return anything. Internal dataset will be updated. 
        """
        self.class_mapping = []
        self.class_index = class_index
        self.num_classes = num_classes
        if not filename.endswith('.csv'):
            message = 'Unable to open \'{0}\'. \
                            Only datasets in CSV format are supported.'
            raise TypeError(message.format(filename))
        with open(filename) as f:
            fr = csv.reader(f)
            headers = next(fr)

            att_values = [[] for i in range(len(headers))]
            instances = []
            try:
                for i in range(probe_instances):
                    inst = next(fr)
                    instances.append(inst)
                    self.class_mapping.append(inst[self.class_index])
                    for j in range(len(headers)):
                        if j == self.class_index:
                            if i == 0:
                                inst[j] == 'myClass'
                            else:
                                inst[j] == 'notMyClass'
                        try:
                            inst[j] = float(inst[j])
                            att_values[j] = None
                        except ValueError:
                            inst[j] = str(inst[j])
                        if isinstance(inst[j], str):
                            if att_values[j] is not None:
                                if inst[j] not in att_values[j]:
                                    att_values[j].append(inst[j])
                            else:
                                message = 'Attribute {0} has \
                                            both Numeric and Nominal values.'
                                raise ValueError(message.format(headers[j]))
            # Tried to probe more instances than there are in the dataset file
            except StopIteration:
                pass

        attributes = []
        att_values[self.class_index] = ['myClass', 'notMyClass']
        for i in range(len(headers)):
            if att_values[i] is None:
                attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
            else:
                attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))

        dataset = Dataset(attributes, class_index)
        
        self.dataset = dataset
        self.class_mapping = np.unique(self.class_mapping)

    def initialize_binary_learners(self, num_wls, 
                                              min_conf=0.00001, max_conf=0.9, 
                                              min_grace=1, max_grace=10,
                                              min_tie=0.001, max_tie=1,
                                              min_weight=10, max_weight=200, 
                                              exp_step_size=1, seed=1):
        ''' Initialize boosted binary learners.
        Args:
            num_wls (int): Number of weak learners
            Other args (float): Range to randomly generate parameters
            exp_step_size (float): Parameter for hedge algorithm
            seed (int): Random seed
        Returns:
            It does not return anything. Generated binary learners are stored in 
            internal variables. 
        '''
        self.binary_learners = [AdaBoostOLM() for i in range(self.num_classes)]

        for bl in self.binary_learners:
            bl.set_dataset(self.dataset)
            bl.set_class_index(self.class_index)
            bl.set_num_classes(2)
            bl.gen_weaklearners(num_wls, min_conf=min_conf, max_conf=max_conf,
                                  min_grace=min_grace, max_grace=max_grace,
                                  min_tie=min_tie, max_tie=max_tie,
                                  min_weight=min_weight, max_weight=max_weight,
                                  seed=seed)
            bl.set_exp_step_size(exp_step_size)

    def get_num_errors(self):
        return self.num_errors

    def get_dataset(self):
        return self.dataset

    def set_exp_step_size(self, exp_step_size):
        for bl in self.binary_learners:
            bl.set_exp_step_size(exp_step_size)

class oneVSallBoost(AdaBoostOLM):
    '''
    AdaBoostOLM with weak learners formed by oneVSall method

    Notation conversion table: 

    v = expert_weights
    alpha =  wl_weights
    sVec = expert_votes
    yHat_t = expert_preds
    C = cost_mat
    '''

    def make_cov_instance(self, X):
        '''Turns a list of covariates into an Instance set to self.datset 
        with None in the location of the class of interest. This is required to 
        pass to a HoeffdingTree so it can make predictions.

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset

        Returns:
            pred_instance (Instance): An Instance with the covariates X and 
                      None in the correct locations

        '''
        inst_values = list(copy.deepcopy(X))
        inst_values.insert(self.class_index, None)

        indices = range(len(inst_values))
        del indices[self.class_index]
        for i in indices:
            if self.dataset.attribute(index=i).type() == 'Nominal':
                inst_values[i] = int(self.binary_dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
            else:
                inst_values[i] = float(inst_values[i])

        pred_instance = Instance(att_values = inst_values)
        pred_instance.set_dataset(self.binary_dataset)
        return pred_instance

    def make_full_instance(self, X, Y):
        '''Makes a complete Instance set to self.dataset with 
        class of interest in correct place

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset
            Y (string): the class of interest corresponding to these covariates.
        
        Returns:
            full_instance (Instance): An Instance with the covariates X and Y 
                            in the correct locations

        '''

        inst_values = list(copy.deepcopy(X))
        inst_values.insert(self.class_index, Y)
        for i in range(len(inst_values)):
            if self.dataset.attribute(index=i).type() == 'Nominal':
                inst_values[i] = int(self.binary_dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
            else:
                inst_values[i] = float(inst_values[i])

        
        full_instance = Instance(att_values=inst_values)
        full_instance.set_dataset(self.binary_dataset)
        return full_instance

    def predict(self, X, verbose=False):
        '''Runs the entire prediction procedure, updating internal tracking 
        of wl_preds and Yhat, and returns the randomly chosen Yhat

        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset
            verbose (bool): If true, the function prints logs. 

        Returns:
            Yhat (string): The final class prediction
        '''

        self.X = X
        pred_inst = self.make_cov_instance(self.X)

        # Initialize values

        expert_votes = np.zeros(self.num_classes)
        expert_votes_mat = np.zeros([self.num_wls, self.num_classes])
        wl_preds = np.zeros(self.num_wls)
        expert_preds = np.zeros(self.num_wls)
        self.cost_mat_diag = np.zeros([self.num_wls, self.num_classes])

        for i in xrange(self.num_wls):
            # Calculate the new cost matrix
            cost_mat = self.compute_cost(expert_votes, i)

            if self.loss == 'zero_one':
                for r in xrange(self.num_classes):
                    self.cost_mat_diag[i,r] = \
                        np.sum(cost_mat[r]) - self.num_classes*cost_mat[r, r]
            else:
                self.cost_mat_diag[i,:] = np.diag(cost_mat)
            
            # Get our new week learner prediction and our new expert prediction

            wls = self.weaklearners[i]
            pred_probs = []
            for wl in wls:
                pred_probs.append(wl.distribution_for_instance(pred_inst)[0])
            pred_probs = np.array(pred_probs)
            pred_probs /= np.sum(pred_probs)
            expected_costs = np.dot(pred_probs, cost_mat) 
            tmp = [x for x in range(self.num_classes) 
                        if expected_costs[x] == min(expected_costs)]
            wl_preds[i] = np.random.choice(tmp)
            
            if verbose is True:
                print i, expected_costs, pred_probs, wl_preds[i]
            
            expert_votes[int(wl_preds[i])] += self.wl_weights[i]
            expert_votes_mat[i,:] = expert_votes
            tmp = [x for x in range(self.num_classes) 
                        if expert_votes[x] == max(expert_votes)]
            expert_preds[i] = np.random.choice(tmp)

            # print(expert_votes)

        final_votes = np.zeros(self.num_classes)
        for i in xrange(self.num_wls):
            final_votes[int(expert_preds[i])] += self.expert_weights[i]

        if self.loss == 'zero_one':
            pred_index = -1
        else:
            tmp = self.expert_weights/sum(self.expert_weights)
            pred_index = np.random.choice(range(self.num_wls), p=tmp)
        self.Yhat_index = int(expert_preds[pred_index])
        self.wl_preds = wl_preds
        self.expert_preds = expert_preds
        
        self.Yhat = self.find_Y(self.Yhat_index)

        return self.Yhat

    def update(self, Y, X=None, verbose=False):
        '''Runs the entire updating procedure, updating interal 
        tracking of wl_weights and expert_weights
        Args:
            X (list): A list of the covariates of the current data point. 
                      Float for numerical, string for categorical. Categorical 
                      data must have been in initial dataset. If not given
                      the last X used for prediction will be used.
            Y (string): The true class
            verbose (bool): If true, the function prints logs. 
        '''

        if X is None:
            X = self.X

        self.X = X
        self.Y = Y
        full_insts = []
        for label in self.class_mapping:
            if Y == label:
                tempY = 'myClass'
            else:
                tempY = 'notMyClass'
            full_insts.append(self.make_full_instance(self.X, tempY))
        self.Y_index = int(self.find_Y_index(Y))
        self.num_data +=1
        self.num_errors += (self.Y_index != self.Yhat_index)

        expert_votes = np.zeros(self.num_classes)
        for i in xrange(self.num_wls):
            alpha = self.wl_weights[i]

            w = self.get_weight(i)

            wls = self.weaklearners[i]
            for j, wl in enumerate(wls):
                full_instance = full_insts[j]
                full_instance.set_weight(w)
                wl.update_classifier(full_instance) 

            if verbose is True:
                print i, w

            # update the quality weights and weighted vote vector
            self.wl_weights[i] = self.update_alpha(expert_votes, i, alpha)
            expert_votes[int(self.wl_preds[i])] += alpha
            self.expert_weights[i] *= \
                np.exp(-(self.expert_preds[i]!=self.Y_index)*self.exp_step_size)

        self.expert_weights = self.expert_weights/sum(self.expert_weights)

    def initialize_dataset(self, filename, class_index, probe_instances=10000):
        """ CODE HERE TAKEN FROM main.py OF HOEFFDINGTREE 
        Open and initialize a dataset in CSV format.
        The CSV file needs to have a header row, from where the attribute 
        names will be read, and a set of instances containing at least one 
        example of each value of all nominal attributes.

        Args:
            filename (str): The name of the dataset file (including filepath).
            class_index (int): The index of the attribute to be set as class.
            probe_instances (int): The number of instances to be used to 
                initialize the nominal attributes. (default 100)

        Returns:
            It does not return anything. Internal dataset will be updated. 
        """
        self.class_index = class_index
        if not filename.endswith('.csv'):
            message = 'Unable to open \'{0}\'. \
                        Only datasets in CSV format are supported.'
            raise TypeError(message.format(filename))
        with open(filename) as f:
            fr = csv.reader(f)
            headers = next(fr)

            att_values = [[] for i in range(len(headers))]
            instances = []
            try:
                for i in range(probe_instances):
                    inst = next(fr)
                    instances.append(inst)
                    for j in range(len(headers)):
                        try:
                            inst[j] = float(inst[j])
                            att_values[j] = None
                        except ValueError:
                            inst[j] = str(inst[j])
                        if isinstance(inst[j], str):
                            if att_values[j] is not None:
                                if inst[j] not in att_values[j]:
                                    att_values[j].append(inst[j])
                            else:
                                message = 'Attribute {0} has \
                                        both Numeric and Nominal values.'
                                raise ValueError(message.format(headers[j]))
            # Tried to probe more instances than there are in the dataset file
            except StopIteration:
                pass

        attributes = []
        for i in range(len(headers)):
            if att_values[i] is None:
                attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
            else:
                attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))

        dataset = Dataset(attributes, class_index)
        self.num_classes = dataset.num_classes()

        if self.loss == 'zero_one':
            self.biased_uniform = \
                        np.ones(self.num_classes)*(1-self.gamma)/self.num_classes
            self.biased_uniform[0] += self.gamma


        self.class_mapping = [dataset.class_attribute().value(i) 
                                for i in xrange(self.num_classes)]
        self.dataset = dataset

        attributes = []
        att_values[self.class_index] = ['myClass', 'notMyClass']
        for i in range(len(headers)):
            if att_values[i] is None:
                attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
            else:
                attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))
        binary_dataset = Dataset(attributes, class_index)
        self.binary_dataset = binary_dataset

    def gen_weaklearners(self, num_wls, min_conf = 0.00001, max_conf = 0.9, 
                                              min_grace = 1, max_grace = 10,
                                              min_tie = 0.001, max_tie = 1,
                                              min_weight = 10, max_weight = 200, 
                                              seed = 1):
        ''' Generate weak learners.
        Args:
            num_wls (int): Number of weak learners
            Other args (float): Range to randomly generate parameters
            seed (int): Random seed
        Returns:
            It does not return anything. Generated weak learners are stored in 
            internal variables. 
        '''

        np.random.seed(seed)
        self.num_wls = num_wls
        self.weaklearners = [[HoeffdingTree() for _2 in xrange(self.num_classes)] 
                                for _1 in range(num_wls)]
        
        min_conf = np.log10(min_conf)
        max_conf = np.log10(max_conf)
        min_tie = np.log10(min_tie)
        max_tie = np.log10(max_tie)

        for wls in self.weaklearners:
            conf = 10**np.random.uniform(low=min_conf, high=max_conf)
            grace = np.random.uniform(low=min_grace, high=max_grace)
            tie = 10**np.random.uniform(low=min_tie, high=max_tie)
            for wl in wls:
                wl._header = self.binary_dataset
                wl.set_split_confidence(conf)
                wl.set_grace_period(grace)
                wl.set_hoeffding_tie_threshold(tie)
        
        self.wl_edges = np.zeros(num_wls)
        self.expert_weights = np.ones(num_wls)/num_wls
        if self.loss == 'zero_one':
            self.wl_weights = np.ones(num_wls)   
        else:
            self.wl_weights = np.zeros(num_wls)
        self.wl_preds = np.zeros(num_wls)
        self.expert_preds = np.zeros(num_wls)

        self.weight_consts = [np.random.uniform(low=min_weight, high=max_weight)
                                for _ in range(num_wls)]
