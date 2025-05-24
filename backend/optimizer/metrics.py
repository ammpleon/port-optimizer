import numpy as np
import pandas as pd
import datetime as dt
from backend.data_grab import DataFetcher
import scipy.optimize as sci_opt



class MeanVariance:
    """This class contains the components of calculating the historical mean return methods"""
    def __init__(self, adjClose_df):
        self.adjClose_df = adjClose_df
        self.n_tickers = len(self.adjClose_df.columns)
        self.log_returns = np.log1p(self.adjClose_df.pct_change())
        self.covMatrix = self.log_returns.cov()
        
        


    #calculating the daily log returns by 252 active trading days
    def calc_meanReturns(self):
        """Calculate the log returns"""

        return self.log_returns.mean()

    def calc_expected_returns(self, weights):

        expected_returns = np.sum(self.calc_meanReturns() * weights)* 252
        
        return expected_returns

    def calc_volatility(self, weights):
        """Calculate the expected volatility"""
        volatility = np.sqrt(
            np.dot(
                weights.T,
                    np.dot(
                        #covariance matrix
                        self.covMatrix * 252,
                    weights       
                    ))
        )

        return volatility

    @staticmethod   
    def annualize_treasury(df):
        rf_df = DataFetcher._fetch_treasury()
        rf_df = rf_df.loc[rf_df.index.intersection(df.index)]
        rf_rate = (1 + rf_df.iloc[1:]['value'].mean()) ** (1/252) - 1

        return rf_rate

class OptimizationMethods(MeanVariance):
    """This class consists of optimization methods for the maximization and minimization problems."""
    def __init__(self, adjClose_df):
        super().__init__(adjClose_df)

        self.init_guess = self.n_tickers * [1./self.n_tickers]
        self.rf_rate = self.annualize_treasury(self.adjClose_df)
        self.bounds = tuple((0, 1) for n_ticker in range(self.n_tickers))
        self.args = (self.calc_meanReturns(), self.covMatrix )


     
    def get_results(self, weights: list) -> np.array:
            """extract the financial metrics (returns, volatility, sharpe ratio)
            from the optimization methods"""
            weights = np.array(weights)

            log_returns = self.log_returns
            expected_returns = np.sum(log_returns.mean() * weights)*252

            volatility = self.calc_volatility(weights)
            #formula is expected returns - risk free rate (annualized) / standard deviation (volatility)
            sharpe_ratio = (expected_returns - self.rf_rate)/volatility

            return np.array([expected_returns, volatility, sharpe_ratio])

    def minimize_volatility(self):

        constraints = ({"type": "eq",
                       "fun": lambda x: np.sum(x) - 1})

        result = sci_opt.minimize(self.calc_volatility,
                                  self.init_guess,
                                  method = "SLSQP",
                                  bounds = self.bounds,
                                  constraints = constraints,
                                  )
        
        optimized_metrics = self.get_results(weights= result.x)

        return result["x"], optimized_metrics 

    def maxmimize_sharpe(self):
        """To maxmimize the sharpe ratio, we have to find the minimal negative sharpe ratio"""
        def neg_sharpe(weights): #Define the objective function for minimizing the negative sharpe

            expected_returns = np.sum(self.calc_meanReturns() * weights) * 252
            volatility = self.calc_volatility(weights)
            #formula for sharpe ratio is expected returns - risk free rate / volatility but we have to plug the minus symbol
            #because we want the negative sharpe
            return -((expected_returns -self.rf_rate)/volatility)
        
        constraints = ({"type": "eq",
                       "fun": lambda x: np.sum(x) - 1})

        result = sci_opt.minimize(neg_sharpe,
                                  self.init_guess,
                                  method = "SLSQP",
                                  bounds = self.bounds,
                                  constraints = constraints,
                                  )
        
        optimized_metrics = self.get_results(weights = result["x"])

       

        return result["x"], optimized_metrics  #Optimized weights and optimized metrics
    

    def efficientOpt(self, returnTarget):
        "This method optimizes the efficient frontier by using the portfolio variance as the objecive function."

        constraints = ({"type": "eq",
                        "fun": lambda x: self.calc_expected_returns(x) - returnTarget},
                        {"type" : "eq",
                         "fun" : lambda x: np.sum(x) - 1})
        
        efficientOpt = sci_opt.minimize(self.calc_volatility,
                                        self.init_guess,
                                        method = "SLSQP",
                                        bounds = self.bounds,
                                        constraints = constraints,
                                        )

        optimized_metrics = self.get_results(weights = efficientOpt.x)

        return efficientOpt, efficientOpt.x, optimized_metrics  #We need to return the scipy's results instead of adding it into a df to ensure
                                               #if the operation was a success.

        
    
    
        



        
        

    
        

