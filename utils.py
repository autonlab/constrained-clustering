import numpy as np

def print_verbose(string, verbose, level = 0, **args):
    if verbose > level:
        print(string, **args)
        
class ConfidenceModel:
    """
        Model which characterized the confidence of the predictions
    """
    
    def __init__(self, model, data_size = None):
        """
            Wrapped the given model
        
            Arguments:
                model {Model} -- Sklearn Model with estimators
                data_size {int} -- If the size of the data to predict is constant 
                    (for memory optimization) (default: None)
        """
        self.model = model

        if data_size:
            self.predictions_init = True
            self.predictions = np.zeros((data_size, model.n_estimators))
        else:
            self.predictions_init = False
    
    def fit(self, data, label):
        return self.model.fit(data, label)
    
    def predict(self, data):
        return self.model.predict(data)
    
    def predict_confidence(self, data):
        """
            Predicts the regressed value for data with confidence
            
            Arguments:
                data {Array} -- Dataset
                
            Returns:
                predicted value, confidence {Array}
        """
        if not self.predictions_init:
            self.predictions =  np.zeros((data.shape[0], len(self.model.estimators_)))
        
        for i, model in enumerate(self.model.estimators_):
            self.predictions[:, i] = self.model.predict(data)

        return self.predictions[:data.shape[0]].mean(1), self.predictions[:data.shape[0]].std(1)