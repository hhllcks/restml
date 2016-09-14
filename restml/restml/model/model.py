from abc import ABCMeta, abstractmethod

class RESTmlModel:
    __metaclass__ = ABCMeta
    
    def __init__(self, modeldata):
        self._modeldata = modeldata
        self._check = self.checkModeldata()
        
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self):
        raise NotImplementedError
    
    @abstractmethod
    def getType(self):
        raise NotImplementedError
    
    def getModeldata(self):
        return self._modeldata
    
    @abstractmethod
    def checkModeldata(self):
        raise NotImplementedError
    
    