
import statsmodels.api
import statsmodels.genmod.families.family
import numpy as np
from sklearn.metrics import r2_score

class GLM(object):
    '''
    A scikit-learn style wrapper for statsmodels.api.GLM.  The purpose of this class is to 
    make generalized linear models compatible with scikit-learn's Pipeline objects.
    
    family : instance of subclass of statsmodels.genmod.families.family.Family
        The family argument determines the distribution family to use for GLM fitting.
    
    xlabels : iterable of strings, optional (empty by default)
        The xlabels argument can be used to assign names to data columns.  This argument is not
        generally needed, as names can be captured automatically from most standard data 
        structures.  If included, must have length n, where n is the number of features.  Note 
        that column order is used to compute term values and make predictions, not column names.  
    '''
    def __init__(self, family, add_constant=True):
        self.family = family
        self.add_constant = add_constant
        
    def _scrub_x(self, X, offset, exposure, **kwargs):
        '''
        Sanitize input predictors and extract column names if appropriate.
        '''
        no_labels = False
        if 'xlabels' not in kwargs and 'xlabels' not in self.__dict__:
            #Try to get xlabels from input data (for example, if X is a pandas DataFrame)
            try:
                self.xlabels = list(X.columns)
            except AttributeError:
                try:
                    self.xlabels = list(X.design_info.column_names)
                except AttributeError:
                    try:
                        self.xlabels = list(X.dtype.names)
                    except TypeError:
                        no_labels = True
        elif 'xlabels' not in self.__dict__:
            self.xlabels = kwargs['xlabels']
        
        #Convert to internally used data type
        X = np.asarray(X,dtype=np.float64)
        m,n = X.shape
        
        if offset is not None:
            offset = np.asarray(offset,dtype=np.float64)
            offset = offset.reshape(offset.shape[0])
        
        if exposure is not None:
            exposure = np.asarray(exposure,dtype=np.float64)
            exposure = exposure.reshape(exposure.shape[0])
        
        #Make up labels if none were found
        if no_labels:
            self.xlabels = ['x'+str(i) for i in range(n)]
        
        return X, offset, exposure
    
    def _scrub(self, X, y, offset, exposure, **kwargs):
        '''
        Sanitize input data.
        '''
        #Check whether X is the output of patsy.dmatrices
        if y is None and type(X) is tuple:
            y, X = X
        
        #Handle X separately
        X, offset, exposure = self._scrub_x(X, offset, exposure, **kwargs)
        
        #Convert y to internally used data type
        y = np.asarray(y,dtype=np.float64)
        y = y.reshape(y.shape[0])
        
        #Make sure dimensions match
        if y.shape[0] != X.shape[0]:
            raise ValueError('X and y do not have compatible dimensions.')
        
        return X, y, offset, exposure
    
    def fit(self, X, y = None, offset = None, exposure = None, xlabels = None):
        '''
        Fit a GLM model to the input data X and y.
        
        
        Parameters
        ----------
        X : array-like, shape = [m, n] where m is the number of samples and n is the number of features
            The training predictors.  The X parameter can be a numpy array, a pandas DataFrame, a patsy 
            DesignMatrix, or a tuple of patsy DesignMatrix objects as output by patsy.dmatrices.
            
        
        y : array-like, optional (default=None), shape = [m] where m is the number of samples
            The training response.  The y parameter can be a numpy array, a pandas DataFrame with one 
            column, a Patsy DesignMatrix, or can be left as None (default) if X was the output of a 
            call to patsy.dmatrices (in which case, X contains the response).
            
        
        xlabels : iterable of strings, optional (default=None)
            Convenient way to set the xlabels parameter while calling fit.  Ignored if None (default).  
            See the GLM class for an explanation of the xlabels parameter.
            
        '''
        
        #Format and label the data
        if xlabels is not None:
            self.set_params(xlabels=xlabels)
        X, y, offset, exposure = self._scrub(X,y,offset,exposure,**self.__dict__)
        
        #Add a constant column
        if self.add_constant:
            X = statsmodels.api.add_constant(X, prepend=True)
        
        #Do the actual work
        model = statsmodels.api.GLM(y, X, self.family, offset=offset, exposure=exposure)
        result = model.fit()
        self.coef_ = result.params
        
        return self
    
    def predict(self, X, offset = None, exposure = None):
        '''
        Predict the response based on the input data X.
        
        
        Parameters 
        ----------
        X : array-like, shape = [m, n] where m is the number of samples and n is the number of features
            The training predictors.  The X parameter can be a numpy array, a pandas DataFrame, or a 
            patsy DesignMatrix.
        '''
        #Format the data
        X, offset, exposure = self._scrub_x(X, offset, exposure)
        
        #Linear transformation
        eta = self.transform(X, offset, exposure)
        
        #Nonlinear transformation
        y_hat = self.family.fitted(eta)
        
        return y_hat
    
    def transform(self, X, offset = None, exposure = None):
        '''
        Perform a linear transformation of X.

        
        Parameters 
        ----------
        X : array-like, shape = [m, n] where m is the number of samples and n is the number of features
            The training predictors.  The X parameter can be a numpy array, a pandas DataFrame, or a 
            patsy DesignMatrix.
        '''
        #Format the data
        X, offset, exposure = self._scrub_x(X, offset, exposure)
        
        #Add a constant column
        if self.add_constant:
            X = statsmodels.api.add_constant(X, prepend=True)
        
        #Compute linear combination
        eta = np.dot(X,self.coef_)
        
        #Apply offset and exposure
        if offset is not None:
            eta += offset
        if exposure is not None:
            eta += np.log(exposure)
        return eta
    
    def score(self, X, y = None, offset = None, exposure = None, xlabels = None):
        X, y, offset, exposure = self._scrub(X,y,offset,exposure,**self.__dict__)
        
        y_pred = self.predict(X, offset=offset, exposure=exposure)
        return r2_score(y, y_pred)
        
    def get_params(self, deep = False):
        return {}
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
    def __str__(self):
        return self.__class__.__name__ + '()'
    
class GLMFamily(GLM):
    family = NotImplemented
    def __init__(self, add_constant=True):
        super(GLMFamily,self).__init__(family=self.__class__.family(), add_constant=add_constant)

class BinomialRegressor(GLMFamily):
    family = statsmodels.genmod.families.family.Binomial

class GammaRegressor(GLMFamily):
    family = statsmodels.genmod.families.family.Gamma
    
class GaussianRegressor(GLMFamily):
    family = statsmodels.genmod.families.family.Gaussian
    
class InverseGaussianRegressor(GLMFamily):
    family = statsmodels.genmod.families.family.InverseGaussian

class NegativeBinomialRegressor(GLMFamily):
    family = statsmodels.genmod.families.family.NegativeBinomial

class PoissonRegressor(GLMFamily):
    family = statsmodels.genmod.families.family.Poisson
#    def fit(self, X, y = None, exposure = None, xlabels = None):
#        '''
#        Fit a GLM model to the input data X and y.
#        
#        
#        Parameters
#        ----------
#        X : array-like, shape = [m, n] where m is the number of samples and n is the number of features
#            The training predictors.  The X parameter can be a numpy array, a pandas DataFrame, a patsy 
#            DesignMatrix, or a tuple of patsy DesignMatrix objects as output by patsy.dmatrices.
#            
#        
#        y : array-like, optional (default=None), shape = [m] where m is the number of samples
#            The training response.  The y parameter can be a numpy array, a pandas DataFrame with one 
#            column, a Patsy DesignMatrix, or can be left as None (default) if X was the output of a 
#            call to patsy.dmatrices (in which case, X contains the response).
#            
#        
#        xlabels : iterable of strings, optional (default=None)
#            Convenient way to set the xlabels parameter while calling fit.  Ignored if None (default).  
#            See the GLM class for an explanation of the xlabels parameter.
#            
#        '''
#        #Format and label the data
#        if xlabels is not None:
#            self.set_params(xlabels=xlabels)
#        X, y = self._scrub(X,y,**self.__dict__)
#        if exposure is not None:
#            exposure = np.asarray(exposure)
#            exposure = exposure.reshape(exposure.shape[0])
#            if exposure.shape != y.shape:
#                raise ValueError('Shape of exposure does not match shape of y.')
#        
#        #Add a constant column
#        if self.add_constant:
#            X = statsmodels.api.add_constant(X, prepend=True)
#        
#        #Do the actual work
#        if exposure is None:
#            model = statsmodels.api.GLM(y, X, self.family)
#        else:
#            model = statsmodels.api.GLM(y, X, self.family, exposure=exposure)
#        result = model.fit()
#        self.coef_ = result.params
#        
#        return self

