from backend.optimizer import MeanVariance 
from backend.data_grab import DataFetcher
import numpy as np
import pandas as pd


class MonteCarlo:

    def __init__(self, adjClose_df):
        self.adjClose_df= adjClose_df

    

    def simulate_portfolio(self, n_sims = 10000):
        """Random Portfolio Generation"""

        num_assets = len(self.adjClose_df.columns)
        weights_array = np.zeros((n_sims, num_assets))
        returns_array = np.zeros(n_sims)
        stdev_array = np.zeros(n_sims)
        sharpe_array = np.zeros(n_sims)
        rf_rate = MeanVariance.annualize_treasury(df =self.adjClose_df)

        mv = MeanVariance(adjClose_df=self.adjClose_df)

        for index in range(n_sims):
            
            random_weights = np.array(np.random.random(num_assets))
            weights = random_weights/sum(random_weights)
            weights_array[index, :] = weights
            returns_array[index] = np.sum(mv.calc_meanReturns() * weights)*252
            stdev_array[index] = mv.calc_volatility(weights)
            #formula for sharpe ratio: expected returns - risk free rate / standard deviation
            sharpe_array[index] = (returns_array[index] - rf_rate)/stdev_array[index]

        #Generate the DataFrame
        weight_columns = [f"Weight_{i}" for i in range(num_assets)]

        simulation_results = pd.DataFrame({
            "Expected Returns": returns_array,
            "Volatility": stdev_array,
            "Sharpe Ratio": sharpe_array,
        **{col: weights_array[:, i]for i, col in enumerate(weight_columns)}})

        new_cols = {}
        for i, ticker_name in enumerate(self.adjClose_df.columns):
            new_cols[f"Weight_{i}"] = ticker_name
        
            
        simulation_results = simulation_results.rename(new_cols, axis = 1)

        return simulation_results

    
         

       