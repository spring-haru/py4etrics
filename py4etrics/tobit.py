"""
Created by Tetsu Haruyama
Last modified 13 July 2019
"""

import warnings
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from py4etrics.base_for_models import GenericLikelihoodModel_TobitTruncreg

class Tobit(GenericLikelihoodModel_TobitTruncreg):
    """
    Create a column in DataFrame (to be used for "cens" below)
           -1: left-censored
            0: uncensored
            1: right-censored

    Method 1:
    Tobit(endog, exog, cens=<COLUMN NAME>, left=<0>, right=<0>).fit()
    endog = dependent variable
    exog = independent variable (add constant if needed)
    cens = see above
    left = the threshold value for left-censoring (default:0)
    right = the threshold value for right-censoring (default:0)

    Method 2:
    formula = 'y ~ 1 + x'
    Tobit.from_formula(formula, cens=<COLUMN NAME>, left=<0>, right=<0>, data=<DATA>).fit()
    """

    def __init__(self, endog, exog, cens, left=None, right=None, **kwds):
        super(Tobit, self).__init__(endog, exog, **kwds)
        self.cens = cens
        if left is None:
            left = 0
        self.left = left
        if right is None:
            right = 0
        self.right = right

    def loglikeobs(self, params): # see _tobit()
#     def loglike(self, params): # see _tobit()
        s = params[-1]
        beta = params[:-1]

        def _tobit(y,x,z,left,right,beta,s):
            if (~np.isin(z,[-1,0,1])).any():
                warnings.warn('\n\n***************************************************\n\n'+
                              'There are values other than [-1,0,1] in a column for cens\n\n'+
                             '***************************************************\n')
            # indicators
            left_on = np.where(z==-1, 1, 0)
            mid_on = np.where(z==0, 1, 0)
            right_on = np.where(z==1, 1, 0)

            Xb = np.dot(x, beta)

            left_mle = left_on * norm.logsf(Xb, loc=left, scale=np.exp(s))
            mid_mle = mid_on * ( -s  + norm.logpdf(y, loc=np.dot(x, beta), scale=np.exp(s)) )
            right_mle = right_on * norm.logcdf(Xb, loc=right, scale=np.exp(s))
        #     return (left_mle+mid_mle+right_mle).sum()  #  loglike()
            return left_mle+mid_mle+right_mle  #  nloglikeobs()

        return _tobit(self.endog, self.exog, self.cens, self.left, self.right, beta, s)

    def fit(self, cov_type='nonrobust', start_params=None, maxiter=10000, maxfun=10000, **kwds):
        # add sigma for summary
        if 'Log(Sigma)' not in self.exog_names:
            self.exog_names.append('Log(Sigma)')
        else:
            pass
        # initial guess
        res = sm.OLS(self.endog, self.exog).fit()
        ols_params = res.params
        ols_sigma = np.log(np.std(res.resid))
        # option
        if start_params == None:
            start_params = np.append(ols_params, ols_sigma)

        res = super(Tobit, self).fit(cov_type=cov_type, start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun, **kwds)
        return res

# EOF
