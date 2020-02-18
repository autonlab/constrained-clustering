import GPy
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
            self.predictions[:, i] = model.predict(data)

        return self.predictions[:data.shape[0]].mean(1), self.predictions[:data.shape[0]].std(1)

class GPModel(ConfidenceModel):
    """
        Gaussian Process Model
    """

    def __init__(self, data_size, n_jobs = 1, ard  = False, kernel = None,
                 num_inducing = 10, mean_function = None, optimize_restarts = 5):
        if kernel is None:
            kernel = GPy.kern.Matern52(data_size, variance=1., ARD=ard)
        self.kernel = kernel
        self.model = None
        self.num_inducing = num_inducing
        self.mean_function = mean_function
        self.optimize_restarts = optimize_restarts

    def _create_model(self, data, label):
        self.model = GPy.models.SparseGPRegression(data, label, kernel=self.kernel, 
                                                   num_inducing=self.num_inducing, 
                                                   mean_function=self.mean_function)
        self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)

    def fit(self, data, label):
        if self.model is None:
            self._create_model(data, label[:, np.newaxis])
        else:
            self.model.set_XY(data, label[:, np.newaxis])
        if self.optimize_restarts == 1:
            self.model.optimize(optimizer = 'bfgs', max_iters = 1000, messages = False, 
                                ipython_notebook = False)
        else:
            self.model.optimize_restarts(num_restarts = self.optimize_restarts, optimizer = 'bfgs', 
                                         max_iters = 1000)

    def predict(self, data, withnoise = True):
        return self.predict_confidence(data, withnoise)[0]

    def predict_confidence(self, data, withnoise = True):
        if data.ndim == 1:
            data = data[None,:]
        prediction, likelihood = self.model.predict(data, full_cov = False, include_likelihood = withnoise)
        likelihood = np.clip(likelihood, 1e-10, np.inf)
        return prediction, np.sqrt(likelihood)