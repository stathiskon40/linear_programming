import pulp
import math

def optimize_portfolio(expected_returns, prices, tickers, budget, min_stocks=3, max_fraction_per_stock=0.4):
    """
    Optimize portfolio selection to maximize expected return while ensuring 
    diversification and staying within budget. Also enforces a maximum fraction 
    of budget on any single stock. Decision variables represent 
    the number of shares to purchase for each stock.

    Args:
        expected_returns (list of float): Expected return per share for each stock.
        prices (list of float): Prices for each stock.
        tickers (list of str): Ticker symbols corresponding to each stock.
        budget (float): Total budget available.
        min_stocks (int): Minimum number of different stocks to invest in.
        max_fraction_per_stock (float): Maximum fraction of budget allowed for any single stock.

    Returns:
        list of dict: Each dict contains 'ticker', 'shares', 'investment', and 'expected_return'.
    """

    n = len(expected_returns)

    # Initialize the LP problem for maximization
    prob = pulp.LpProblem("Portfolio_Optimization", pulp.LpMaximize)

    # Decision variables:
    # x_i: number of shares to purchase for stock i (integer)
    # y_i: binary variable indicating if stock i is selected
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(n)]
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

    # Big M for linking constraints, one for each stock
    M = [math.floor(budget / prices[i]) for i in range(n)]

    # Objective: Maximize total expected return across all shares purchased
    prob += pulp.lpSum(expected_returns[i] * x[i] for i in range(n)), "Total_Expected_Return"

    # Constraint: Total cost of shares must not exceed the budget
    prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= budget, "Budget_Constraint"

    # Linking constraints: Ensure y_i = 0 implies x_i = 0
    for i in range(n):
        prob += x[i] <= M[i] * y[i], f"Linking_{i}"

    # Diversification constraint: At least min_stocks must be selected
    prob += pulp.lpSum(y[i] for i in range(n)) >= min_stocks, "Minimum_Diversification"

    # Per-stock spending limits: No more than max_fraction_per_stock of budget on any single stock
    for i in range(n):
        prob += prices[i] * x[i] <= max_fraction_per_stock * budget, f"MaxFraction_{i}"

    # Solve the optimization problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check solver status
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("No optimal solution found.")
        return []

    # Collect results: list of dicts with ticker, shares, investment, and expected return
    results = []
    for i in range(n):
        shares = x[i].varValue
        if shares is not None and shares > 0:
            investment_amount = shares * prices[i]
            expected_ret = shares * expected_returns[i]
            results.append({
                'ticker': tickers[i],
                'shares': int(shares),
                'investment': investment_amount,
                'expected_return': expected_ret
            })

    return results
