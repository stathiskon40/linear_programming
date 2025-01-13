import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
import pulp

class DataProcessor:
    def __init__(self, df_current: pd.DataFrame, df_predicted: pd.DataFrame):
        self.df_current = df_current
        self.df_predicted = df_predicted
        self.merged_df = None

    def validate_dataframes(self):
        required_current = {'ticker', 'current_price'}
        required_predicted = {'ticker', 'predicted_price'}
        
        if not required_current.issubset(self.df_current.columns):
            missing = required_current - set(self.df_current.columns)
            raise ValueError(f"df_current is missing columns: {missing}")
        
        if not required_predicted.issubset(self.df_predicted.columns):
            missing = required_predicted - set(self.df_predicted.columns)
            raise ValueError(f"df_predicted is missing columns: {missing}")

    def merge_dataframes(self) -> pd.DataFrame:
        self.validate_dataframes()
        self.merged_df = pd.merge(
            self.df_current, 
            self.df_predicted, 
            on='ticker', 
            how='inner', 
            suffixes=('_current', '_predicted')
        )
        self.merged_df['expected_return'] = self.merged_df['predicted_price'] - self.merged_df['current_price']
        return self.merged_df

    def calculate_difference(self) -> pd.DataFrame:
        if self.merged_df is None:
            self.merge_dataframes()
        self.merged_df['price_difference'] = self.merged_df['predicted_price'] - self.merged_df['current_price']
        return self.merged_df

    def optimize_portfolio(self, budget: float):
        """
        Optimize the portfolio to maximize expected returns within the budget.

        Args:
            budget (float): Total budget available for investment.

        Returns:
            dict: Optimization results including selected stocks and their quantities.
        """
        if self.merged_df is None:
            self.merge_dataframes()

        # Prepare data for optimization
        stocks = self.merged_df['ticker'].tolist()
        returns = self.merged_df.set_index('ticker')['expected_return'].to_dict()
        prices = self.merged_df.set_index('ticker')['current_price'].to_dict()

        # Initialize the LP problem
        problem = LpProblem("Portfolio_Optimization", LpMaximize)

        # Define decision variables: number of shares to buy for each stock
        # Assuming you can buy fractional shares; set cat='Continuous'
        # If you need integer shares, set cat='Integer'
        x = LpVariable.dicts("Shares", stocks, lowBound=0, cat='Integer')

        # Objective: Maximize total expected return
        problem += lpSum([returns[i] * x[i] for i in stocks]), "Total Expected Return"

        # Constraint: Total cost does not exceed budget
        problem += lpSum([prices[i] * x[i] for i in stocks]) <= budget, "Total Budget"

        # (Optional) Additional constraints can be added here
        # For example, limit the number of different stocks in the portfolio

        # Solve the problem
        problem.solve(pulp.PULP_CBC_CMD(msg=False))

        # Check the status of the solution
        status = LpStatus[problem.status]
        if status != 'Optimal':
            raise Exception(f"Optimization failed with status: {status}")

        # Extract the results
        investment = {stock: x[stock].varValue for stock in stocks if x[stock].varValue > 0}

        total_invested = sum(prices[stock] * qty for stock, qty in investment.items())
        total_return = sum(returns[stock] * qty for stock, qty in investment.items())

        return {
            'status': status,
            'investment': investment,
            'total_invested': total_invested,
            'total_expected_return': total_return
        }
