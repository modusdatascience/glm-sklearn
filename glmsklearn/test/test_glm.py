from nose.tools import assert_equal, assert_not_equal, assert_true, assert_false, \
    assert_almost_equal, assert_list_equal
import numpy as np
from glmsklearn import BinomialRegressor, GammaRegressor, GaussianRegressor, \
    InverseGaussianRegressor, NegativeBinomialRegressor, PoissonRegressor
from statsmodels.genmod.families.family import Binomial, Gamma, Gaussian,\
    InverseGaussian, NegativeBinomial, Poisson
from sklearn.pipeline import Pipeline
from sklearn.decomposition.pca import PCA



class TestGlm():
    def __init__(self):
        #Generate some data
        np.random.seed(1)
        self.X = np.random.normal(scale=.5,size=(100,10))**2
        self.beta = np.random.normal(scale=.5,size=10)**2
        self.eta = np.dot(self.X, self.beta) + .1*np.random.normal(size=100)
        
    def test_binomial(self):
        model = BinomialRegressor()
        y = Binomial().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .9)
    
    def test_gamma(self):
        model = GammaRegressor()
        y = Gamma().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .58)
    
    def test_gaussian(self):
        model = GaussianRegressor()
        y = Gaussian().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .9)
        
    def test_inverse_gaussian(self):
        model = InverseGaussianRegressor()
        y = InverseGaussian().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .6)
        
    def test_negative_binomial(self):
        model = NegativeBinomialRegressor()
        y = NegativeBinomial().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .9)
        
    def test_poisson(self):
        model = PoissonRegressor()
        y = Poisson().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .9)
        
    def test_poisson_exposure(self):
        model = PoissonRegressor()
        exposure = np.random.exponential(scale=10, size=100)
        y = Poisson().fitted(self.eta + np.log(exposure))
        model.fit(self.X, y, exposure=exposure)
        y_hat = model.predict(self.X, exposure=exposure)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .9)
    
    def test_with_pipeline(self):
        model = Pipeline([('PCA',PCA()), ('Poisson',PoissonRegressor())])
        y = Poisson().fitted(self.eta)
        model.fit(self.X, y)
        y_hat = model.predict(self.X)
        diff = y_hat - y
        rsq = 1 - np.mean(diff**2) / np.mean((y-np.mean(y))**2)
        assert_true(rsq > .9)
        assert_equal(str(model), '''Pipeline(PCA=PCA(copy=True, n_components=None, whiten=False), PCA__copy=True,
     PCA__n_components=None, PCA__whiten=False, Poisson=PoissonRegressor())''')
        
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])
    
    