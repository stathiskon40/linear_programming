import pulp

def optimize_portfolio_with_risk(
    tickers,
    current_prices,        # dict: {ticker: current_price}
    predicted_returns,     # dict: {ticker: predicted_return_in_decimal}, e.g. 0.02 for 2%
    risks,                 # dict: {ticker: risk_score}, e.g. 5.0 or 0.05
    budget=10000,          # total money available
    lambda_penalty=0.5,    # how much to penalize risk
    fully_invest=False,    # if True => sum(...) == budget, else <= budget
    sector_constraints=None,
    max_sector_allocation=None,
    solver=None,
    verbose=False
):
    """
    Optimize portfolio allocation for a fixed dollar budget.

    Decision Variables:
    ------------------
      x_t >= 0 : number of shares of ticker t

    Objective:
    ----------
      Maximize sum( predicted_profit ) - lambda * sum( risk_penalty )

      where 
        predicted_profit(t) = current_prices[t] * x_t * predicted_returns[t]
        risk_penalty(t)     = current_prices[t] * x_t * risks[t]

    Constraints:
    ------------
      1) Budget constraint: sum( x_t * current_prices[t] ) <= or == budget
      2) Sector constraints (optional)

    Parameters:
    -----------
    tickers : list
        List of ticker symbols.

    current_prices : dict
        Mapping ticker -> current market price. { 'AAPL': 150.0, 'TSLA': 220.0, ... }

    predicted_returns : dict
        Mapping ticker -> predicted return in decimal form, e.g. 0.02 = 2% return.

    risks : dict
        Mapping ticker -> risk score or volatility measure, e.g. 5.0 or 0.05.

    budget : float
        Total dollar budget available.

    lambda_penalty : float
        Risk-aversion coefficient. Higher => invests in lower-risk assets.

    fully_invest : bool
        If True, sum of all (x_t * current_price[t]) == budget (fully invest).
        If False, sum(...) <= budget (allow leftover cash).

    sector_constraints : dict, optional
        e.g. {
            "Tech": ["AAPL", "MSFT", "GOOG"],
            "Auto": ["TSLA", "F"]
        }

    max_sector_allocation : dict, optional
        e.g. {
            "Tech": 0.60,   # up to 60% of the budget
            "Auto": 0.30
        }

    solver : pulp solver, optional
        A PuLP solver instance (e.g. pulp.PULP_CBC_CMD(msg=0)) if you want custom config.

    verbose : bool
        If True, prints solver messages.

    Returns:
    --------
    allocations_result : dict
        Dictionary {ticker: number_of_shares_to_buy}.

    status : str
        LP solution status (e.g. 'Optimal', 'Infeasible', etc.).

    Example:
    --------
        tickers = ["AAPL", "TSLA", "GOOG"]
        current_prices = {"AAPL": 150, "TSLA": 220, "GOOG": 100}
        predicted_returns = {"AAPL": 0.02, "TSLA": 0.03, "GOOG": 0.01}
        risks = {"AAPL": 5, "TSLA": 8, "GOOG": 3}
        
        sector_constraints = {
            "Tech": ["AAPL", "GOOG"],
            "Auto": ["TSLA"]
        }
        
        max_sector_allocation = {
            "Tech": 0.60,  # up to 60% of the total budget in Tech
            "Auto": 0.30,  # up to 30% in Auto
        }

        alloc, status = optimize_portfolio_with_budget(
            tickers,
            current_prices,
            predicted_returns,
            risks,
            budget=10000,
            lambda_penalty=0.5,
            fully_invest=False,
            sector_constraints=sector_constraints,
            max_sector_allocation=max_sector_allocation
        )

        print("Allocations:", alloc)
        print("Status:", status)
    """

    # --------------------------------------
    # 1) Define the LP problem
    # --------------------------------------
    prob = pulp.LpProblem("Portfolio_Optimization_With_Budget", pulp.LpMaximize)

    # Check dictionaries
    for t in tickers:
        if t not in current_prices:
            raise ValueError(f"Ticker {t} not in current_prices.")
        if t not in predicted_returns:
            raise ValueError(f"Ticker {t} not in predicted_returns.")
        if t not in risks:
            raise ValueError(f"Ticker {t} not in risks.")

    # --------------------------------------
    # 2) Decision Variables
    #    x[t] = number of shares of ticker t (can be fractional if broker allows)
    # --------------------------------------
    x = {
        t: pulp.LpVariable(f"x_{t}", lowBound=0, cat=pulp.LpInteger)
        for t in tickers
    }

    # --------------------------------------
    # 3) Objective Function
    #
    #    Maximize: ∑ ( current_price[t]* x[t] * predicted_returns[t] )
    #              - λ ∑ ( current_price[t]* x[t] * risks[t] )
    #
    #    =>  ∑ ( x[t] * (current_price[t]*predicted_returns[t] ) )
    #         - λ ∑ ( x[t] * (current_price[t]*risks[t]) )
    #
    #    =>  ∑ ( x[t] * [ current_price[t]*(predicted_returns[t] - λ*risk[t]) ] )
    # --------------------------------------
    profit_terms = [
        x[t] * (current_prices[t] * predicted_returns[t]) for t in tickers
    ]
    risk_penalty_terms = [
        x[t] * (current_prices[t] * risks[t]) for t in tickers
    ]
    prob += pulp.lpSum(profit_terms) - lambda_penalty * pulp.lpSum(risk_penalty_terms), \
            "Maximize_Profit_Minus_Risk"

    # --------------------------------------
    # 4) Budget Constraint
    #    sum( x[t]*current_prices[t] ) <= (or ==) budget
    # --------------------------------------
    if fully_invest:
        prob += (
            pulp.lpSum([x[t] * current_prices[t] for t in tickers]) == budget,
            "BudgetConstraint_FullyInvested"
        )
    else:
        prob += (
            pulp.lpSum([x[t] * current_prices[t] for t in tickers]) <= budget,
            "BudgetConstraint_NotFullyInvested"
        )

    # --------------------------------------
    # 5) Sector Constraints (Optional)
    #    e.g. sum( cost of Tech tickers ) <= 0.6 * budget
    # --------------------------------------
    if sector_constraints and max_sector_allocation:
        for sector, sector_tickers in sector_constraints.items():
            if sector in max_sector_allocation:
                max_frac = max_sector_allocation[sector]
                prob += (
                    pulp.lpSum([x[t] * current_prices[t] for t in sector_tickers]) 
                    <= max_frac * budget,
                    f"Max_{sector}_Allocation"
                )

    # --------------------------------------
    # 6) Solve the LP
    # --------------------------------------
    if solver is None:
        prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose),timeLimit=5))
    else:
        prob.solve(solver)

    # --------------------------------------
    # 7) Retrieve Results
    # --------------------------------------
    status = pulp.LpStatus[prob.status]
    allocations_result = {
        t: x[t].varValue for t in tickers
    }

    return allocations_result, status
