import scipy
import scipy.stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
'''
erlang is throwing errors
'''
class Distribution(object):
    
    def __init__(self,dist_names_list = [],test=True):
        
        self.dist_names = ["norm", "exponweib", "weibull_max", "weibull_min",
        "pareto", "genextreme","lognorm","expon", "beta", "cauchy", "chi",
        "cosine", "gamma", "logistic", "lomax", "maxwell", "chi2",
        "pearson3", "powerlaw", "rdist", "uniform", "vonmises", "wald", "wrapcauchy","erlang"]
        
        if test:
            self.dist_names = ["norm"]
        # removed distributions ["erlang"]
        # self.dist_names = ["norm"]
        self.dist_results = []
        self.params = {}
        
        self.DistributionName = ""
        self.PValue = 0
        self.Param = None
        
        self.isFitted = False
        
        
    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            # print("Running "+ dist_name)
            dist = getattr(scipy.stats, dist_name)
            try:
                param = dist.fit(y)
                self.params[dist_name] = param
                #Applying the Kolmogorov-Smirnov test
                D, p = scipy.stats.kstest(y, dist_name, args=param);
                self.dist_results.append((dist_name,p))
            except:
                print('encountered an error')

        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results,key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p
        
        self.isFitted = True
        # return self.DistributionName,self.PValue
    
    def Random(self, n = 1):
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            #initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError('Must first run the Fit method.')
            
    def Plot(self,y, show=False):
        x = self.Random(n=len(y))
        if show:
            plt.hist(x, alpha=0.5, label='Fitted')
            plt.hist(y, alpha=0.5, label='Actual')
            plt.legend(loc='upper right')
            plt.show()
        return x, self.DistributionName, self.PValue
