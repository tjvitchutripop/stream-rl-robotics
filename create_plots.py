import plotly
import pandas as pd

# Load data from csv
data = pd.read_csv("data/adaptiveobgdcbp-brokenleg.csv")

# Extract relevant columns
timestep = data["Step"]
success_rate = data["AnymalC-Reach-v1_seed0_hidden_size256_lr1_gamma0.99_lamda0.8_entropy0.01 - eval/success_rate"]

# Divide timestep by 2
timestep = timestep / 2

# Plot success rate vs step using plotly
fig = plotly.graph_objs.Figure()
fig.add_trace(plotly.graph_objs.Scatter(x=timestep, y=success_rate, mode='lines', name='Success Rate'))
fig.update_layout(title='Success Rate Over Time', xaxis_title='Step', yaxis_title='Success Rate')
fig.write_image("figures/adaptiveobgdcbp-brokenleg-success-rate.png")