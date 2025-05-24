import numpy as np
import pandas as pd
import plotly.graph_objects as go 
from backend.optimizer import MeanVariance as mv, OptimizationMethods as om
from backend.simulations import MonteCarlo



def efficientFrontier(data):

    
    mc_results = MonteCarlo(data).simulate_portfolio()
    opt_m = om(data)

    maximized_sharpe = opt_m.maxmimize_sharpe()[1]

    minimized_variance = opt_m.minimize_volatility()[1]

    target_returns = np.linspace(minimized_variance[0], maximized_sharpe[0], 20)
    success_opt = []
    efficient_list = []
    for target in target_returns:
        result = opt_m.efficientOpt(target)
        if result[0]["success"] == np.True_:
            efficient_list.append(result[2][1])
            success_opt.append(target)
        else:
            continue
    
    maxSharpeRatio = go.Scatter(
    name = "Maximum Sharpe Ratio",
    mode = "markers",
    x = [maximized_sharpe[1]],
    y =[maximized_sharpe[0]],
    marker = dict(color = "red", size = 16, line = dict(width = 3, color = "black"), symbol = "star")
)

    minVol = go.Scatter(
    name = "Minimum Volatility",
    mode = "markers",
    x = [f"{round(minimized_variance[1],2)}"],
    y = [f"{round(minimized_variance[0],2)}"],
    marker = dict(color = "green", size = 16, line = dict(width = 3, color = "black"), symbol = "diamond")
    )

    ef_curve = go.Scatter(
    name = "Efficient Frontier",
    mode = "lines",
    x = [n for n in efficient_list],
    y = [target  for target in success_opt],
    line = dict(color = "black", width = 4, dash = "dashdot"),
    hovertemplate=
        "Volatility: %{x:.2%}<br>"
        "Expected Return: %{y:.2%}<extra></extra>"

    )



    mc_scatter = go.Scatter(
    name = "Mone Carlo Simulations",
    mode = "markers",
    x = mc_results["Volatility"],
    y = mc_results["Expected Returns"],
    marker = dict(color = mc_results["Sharpe Ratio"],
                colorscale = "blues",showscale=True,
        cmin=min(mc_results["Sharpe Ratio"]),
        cmax=max(mc_results["Sharpe Ratio"]),
        colorbar=dict(title="Sharpe Ratio")),
    hovertemplate=
        "Volatility: %{x:.2%}<br>"  # Display as percentage
        "Expected Return: %{y:.2%}<br>"
        "Sharpe Ratio: %{marker.color:.2f}<extra></extra>"  # Use marker color (Sharpe Ratio)


    )


    data = [maxSharpeRatio, minVol, ef_curve, mc_scatter]

    layout = go.Layout(
    title = "Portfolio Optimization for efficient Frontier",
    yaxis = dict(title = "Annualized return", 
                 tickformat = ".2%",
                 gridcolor = "rgba(255, 255, 255, 0.8)",
                 zerolinecolor = "rgba(150, 150, 150, 0.5)"
                 ),

    xaxis = dict(title = "Annualized volatility",
                 tickformat = ".2%",
                 gridcolor = "rgba(200, 200, 200, 0.5)",
                 zerolinecolor = "rgba(150, 150, 150, 0.5)"
                 ),
    showlegend = True,
    plot_bgcolor= "#f9f9f9",
    paper_bgcolor = "#ffffff",
    legend = dict(
    x = 0.75, y = 0,
    traceorder = "normal",
    bgcolor = "rgba(255, 255, 255, 0.8)",
    bordercolor = "black",
    borderwidth = 2),
    width = 800,
    height = 600
    )



    fig  = go.Figure(data = data, layout = layout)

    fig.show()

    
