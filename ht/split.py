from abc import ABCMeta, abstractmethod

ABCMeta = ABCMeta('ABCMeta', (object,), {}) # compatible with Python 2 *and* 3 

class Split(ABCMeta):
    """Base for classes that handle splitting (UnivariateNominaMultiwaySplit 
    and UnivariateNumericBinarySplit)."""
    def __init__(self):
        self._split_att_names = []

    @abstractmethod
    def branch_for_instance(self, instance):
        pass

    @abstractmethod
    def condition_for_branch(self, branch):
        pass

    def split_attributes(self):
        return self._split_att_names