from distutils.core import setup
    
#Create a dictionary of arguments for setup
setup_args = {'name':'glm-sklearn',
    'version':'0.1.0',
    'author':'Jason Rudy',
    'author_email':'jcrudy@gmail.com',
    'packages':['glmsklearn','glmsklearn.test'],
    'license':'LICENSE.txt',
    'description':'Scikit-learn style wrappers for statsmodels GLM.',
    'long_description':open('README.md','r').read(),
    'py_modules' : ['glmsklearn.glm','glmsklearn.test.test_glm'],
    'classifiers' : ['Development Status :: 3 - Alpha'],
    'requires':['numpy','statsmodels','sklearn']} 

#Finally
setup(**setup_args)
