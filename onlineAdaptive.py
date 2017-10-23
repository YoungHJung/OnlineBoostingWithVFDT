import csv
import numpy as np
import copy
from hoeffdingtree import *

class AdaBoostOLM:
    '''
    Main class for Online Multiclass AdaBoost algorithm using VFDT.

    Notation conversion table: 

    v = expert_weights
    alpha =  wl_weights
    sVec = expert_votes
    yHat_t = expert_preds
    C = cost_mat

    '''

    def __init__(self, loss='logistic', gamma=0.1):
        '''
        The kwarg loss can take values of 'logistic', 'zero_one', or 'exp'. 
        'zero_one' option corresponds to OnlineMBBM. 

        The value gamma becomes meaningful only when the loss is 'zero_one'. 
        '''
        # Initializing computational elements of the algorithm
        self.num_wls = None
        self.num_classes = None
        self.num_data = 0
        self.dataset = None
        self.class_index = None
        self.num_errors = 0
        self.exp_step_size = 1
        self.loss = loss
        self.gamma = gamma
        self.M = 100

        if self.loss == 'zero_one':
            self.potentials = {}

        self.wl_edges = None
        self.weaklearners = None
        self.expert_weights = None
        self.wl_weights = None
        self.wl_preds = None
        self.expert_preds = None
        self.cost_mat_diag = None

        # Initializing data states
        self.X = None
        self.Yhat_index = None
        self.Y_index = None
        self.Yhat = None
        self.Y = None
        self.pred_conf = None

    ########################################################################

    # Helper functions

    def expit_diff(self, x, y):
        '''Calculates the logistic (expit) difference between two numbers
        Args:
            x (float): positive value
            y (float): negative value
        Returns:
            value (float): the expit difference
        '''
        value = 1/(1 + np.exp(x - y))
        return value

    def exp_diff(self, x, y):
        '''Calculates the exponential of difference between two numbers
        Args:
            x (float): positive value
            y (float): negative value
        Returns:
            value (float): the exponential difference
        '''
        value = np.exp(y - x)
        return value

    def mc_potential(self, t, b, s):
        '''Approximate potential via Monte Carlo simulation
        Arbs:
            t (int)     : number of weak learners until final decision
            b (list)    : baseline distribution
            s (list)    : current state
        Returns:
            potential value (float)
        '''
        k = len(b)
        r = 0
        cnt = 0
        for _ in xrange(self.M):
            x = np.random.multinomial(t, b)
            x = x + s
            tmp = x[r]
            x[r] = 0
            if tmp <= np.max(x):
                cnt += 1
        return float(cnt) / self.M

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
                inst_values[i] = int(self.dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
            else:
                inst_values[i] = float(inst_values[i])

        pred_instance = Instance(att_values = inst_values)
        pred_instance.set_dataset(self.dataset)
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
                inst_values[i] = int(self.dataset.attribute(index=i)
                    .index_of_value(str(inst_values[i])))
            else:
                inst_values[i] = float(inst_values[i])

        
        full_instance = Instance(att_values=inst_values)
        full_instance.set_dataset(self.dataset)
        return full_instance

    def find_Y(self, Y_index):
        '''Get class string from its index
        Args:
            Y_index (int): The index of Y
        Returns:
            Y (string): The class of Y
        '''

        Y = self.dataset.attribute(index=self.class_index).value(Y_index)
        return Y

    def find_Y_index(self, Y):
        '''Get class index from its string
        Args:
            Y (string): The class of Y
        Returns:
            Y_index (int): The index of Y
        '''

        Y_index = int(self.dataset.attribute(index=self.class_index)
                    .index_of_value(Y))
        return Y_index

    ########################################################################

    def get_potential(self, r, n, s):
        '''Compute potential
        Args:
            r (int): True label index
            n (int): Number of weak learners until final decision
            s (list): Current state
        Returns:
            (float) potential function
        '''
        new_s = list(s)
        new_s[r] = -np.inf
        new_s.sort()
        new_s[0] = s[r]

        key = (n, tuple(new_s))
        if key not in self.potentials:
            value = self.mc_potential(n, self.biased_uniform, new_s)
            self.potentials[key] = value
        return self.potentials[key]

    def compute_cost(self, s, i):
        ''' Compute cost matrix
        Args:
            s (list): Current state
            i (int): Weak learner index
        Return:
            (numpy.ndarray) Cost matrix
        '''
        k = self.num_classes
            
        if self.loss == 'logistic':
            ret = self.expit_diff(np.array(s)[:,None], np.array(s)[None,:] ) 
            ret = ret * (np.ones([k, k]) - np.eye(k))
            ret = ret - (np.dot(ret, np.ones(k)) * np.eye(k))
            return ret
        elif self.loss == 'exp':
            ret = self.exp_diff(np.array(s)[:,None], np.array(s)[None,:] ) 
            ret = ret * (np.ones([k, k]) - np.eye(k))
            ret = ret - (np.dot(ret, np.ones(k)) * np.eye(k))
            return ret
        else:
            ret = np.zeros((k, k))
            for r in xrange(k):
                for l in xrange(k):
                    e = np.zeros(k)
                    e[l] = 1
                    ret[r, l] = self.get_potential(r, self.num_wls-i, s+e)
            return ret

    def get_grad(self, s, i, alpha):
        ''' Compute gradient for differnt losses
        Args:
            s (list): Current state
            i (int): Weak learner index
            alpha (float): Weight
        Return:
            (float): Gradient
        '''
        if self.loss == 'logistic':
            if self.wl_preds[i] == self.Y_index:
                tmp_zeroer = np.ones(self.num_classes)
                tmp_zeroer[self.Y_index] = 0
                tmp = -1/ (1 + np.exp(s - (alpha + s[self.Y_index])))
                ret = sum(tmp_zeroer * tmp)
            else:
                tmp = s[self.Y_index] - s[int(self.wl_preds[i])] - alpha
                ret = 1/(1 + np.exp(tmp))
            return ret
        elif self.loss == 'exp':
            if self.wl_preds[i] == self.Y_index:
                tmp_zeroer = np.ones(self.num_classes)
                tmp_zeroer[self.Y_index] = 0
                tmp = np.exp(s - (alpha + s[self.Y_index]))
                ret = sum(tmp_zeroer * tmp)
            else:
                tmp = s[int(self.wl_preds[i])] + alpha - s[self.Y_index]
                ret = np.exp(tmp)
            return ret
        else:
            # Can never reach this case
            return    

    def get_lr(self, i):
        ''' Get learning rate
        Args:
            i (int): Weak learner index
        Return:
            (float): Learning rate
        '''
        if self.loss == 'zero_one':
            return 1
        else:
            ret = 2*np.sqrt(2)/((self.num_classes-1)*np.sqrt(self.num_data))
            if self.loss == 'logistic':
                return ret
            else:
                return ret * np.exp(-i)
        
    def update_alpha(self, s, i, alpha):
        ''' Update the weight alpha
        Args:
            s (list): Current state
            i (int): Weak learner index
            alpha (float): Weight
        Return:
            (float): updated alpha
        '''
        if self.loss == 'zero_one':
            return 1
        else:
            grad = self.get_grad(s, i, alpha)
            lr = self.get_lr(i)
            return max(-2, min(2, alpha - lr*grad))

    def get_weight(self, i):
        ''' Compute sample weight
        Args:
            i (int): Weak learner index
        Return:
            (float): Sample weight
        '''
        if self.loss == 'logistic':
            if i == 0:
                return 5
            const = self.weight_consts[i]
            ret = -const * self.cost_mat_diag[i,self.Y_index]/(self.num_classes-1)
        elif self.loss == 'exp':
            ret = -self.cost_mat_diag[i,self.Y_index]/(self.num_classes-1)
            N = np.exp(0.2*np.sqrt(i))
            ret = 10 * ret / N
        else:
            const = self.weight_consts[i]
            ret = self.cost_mat_diag[i,self.Y_index]/(self.num_classes-1)
            ret = 0.1 * const * ret
        return max(1e-10, ret)

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

            pred_probs = self.weaklearners[i].distribution_for_instance(pred_inst)
            pred_probs = np.array(pred_probs)
            expected_costs = np.dot(pred_probs, cost_mat) 
            tmp = [x for x in range(self.num_classes) 
                        if expected_costs[x] == min(expected_costs)]
            wl_preds[i] = np.random.choice(tmp)
            if verbose is True:
                print i, expected_costs, pred_probs, wl_preds[i]
            expert_votes[int(wl_preds[i])] = \
                    self.wl_weights[i] + expert_votes[int(wl_preds[i])]
            expert_votes_mat[i,:] = expert_votes
            tmp = [x for x in range(self.num_classes) 
                        if expert_votes[x] == max(expert_votes)]
            expert_preds[i] = np.random.choice(tmp)

        final_votes = np.zeros(self.num_classes)
        for i in xrange(self.num_wls):
            final_votes[int(expert_preds[i])] += self.expert_weights[i]

        if self.loss == 'zero_one':
            pred_index = -1
        else:
            tmp = self.expert_weights/sum(self.expert_weights)
            pred_index = np.random.choice(range(self.num_wls), p = tmp)
        self.Yhat_index = int(expert_preds[pred_index])
        self.wl_preds = wl_preds
        self.expert_preds = expert_preds
        self.pred_conf = final_votes

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
        full_inst = self.make_full_instance(self.X, self.Y)
        self.Y_index = int(self.find_Y_index(Y))
        self.num_data +=1
        self.num_errors = self.num_errors + (self.Y_index != self.Yhat_index)

        expert_votes = np.zeros(self.num_classes)
        for i in xrange(self.num_wls):
            alpha = self.wl_weights[i]

            w = self.get_weight(i) 
            full_inst.set_weight(w)
            self.weaklearners[i].update_classifier(full_inst) 

            if verbose is True:
                print i, w

            # updating the quality weights and weighted vote vector
            self.wl_weights[i] = \
                                self.update_alpha(expert_votes, i, alpha)
            expert_votes[int(self.wl_preds[i])] += alpha
            self.expert_weights[i] *= \
              np.exp(-(self.expert_preds[i]!=self.Y_index) * self.exp_step_size)

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
            message = 'Unable to open \'{0}\'. Only datasets in \
                CSV format are supported.'
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
                                raise ValueError(
                                    'Attribute {0} has both Numeric and Nominal values.'
                                    .format(headers[j]))
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

        self.dataset = dataset

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
        self.weaklearners = [HoeffdingTree() for i in range(num_wls)]

        min_conf = np.log10(min_conf)
        max_conf = np.log10(max_conf)
        min_tie = np.log10(min_tie)
        max_tie = np.log10(max_tie)

        for wl in self.weaklearners:
            wl._header = self.dataset
            conf = 10 ** np.random.uniform(low=min_conf, high=max_conf)
            wl.set_split_confidence(conf)
            grace = np.random.uniform(low=min_grace, high=max_grace)
            wl.set_grace_period(grace)
            tie = 10**np.random.uniform(low=min_tie, high=max_tie)
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

    def get_num_errors(self):
        return self.num_errors

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_num_wls(self, n):
        self.num_wls = n

    def set_class_index(self, class_index):
        self.class_index = class_index

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_exp_step_size(self, exp_step_size):
        self.exp_step_size = exp_step_size


