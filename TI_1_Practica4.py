import panel as pn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functions import bm_2d, crw_2d, lf_2d, get_mean_squared_displacement, get_path_length

# Global variables
trajectory_df = pd.DataFrame()
fig_trajectory = go.Figure()

# Panel items
trajectory_type = pn.widgets.RadioButtonGroup(name='Botones', options=['BM', 'CRW', 'LF'])
number_of_steps_input = pn.widgets.IntInput(name='Number of steps', start=1, end=10000, value=100, width=100)
speed_input = pn.widgets.FloatInput(name='Speed', start=1, end=100, value=6, width=100, step=1)
start_x_pos_input = pn.widgets.FloatInput(name='Start X pos', start=0, end=100, value=0, width=100)
start_y_pos_input = pn.widgets.FloatInput(name='Start Y pos', start=0, end=100, value=0, width=100)

cauchy_coefficient_input = pn.widgets.FloatInput(name='Cauchy Coefficient', start=0.1, end=0.9, value=0.1, step=0.1, width=100)

alpha_input = pn.widgets.FloatInput(name='Alpha', start=0.1, end=3, value=1, step=0.1, width=100)
beta_input = pn.widgets.FloatInput(name='Beta', start=0.1, end=1.0, value=1, step=0.1, width=100)
loc_input = pn.widgets.FloatInput(name='Loc', start=0, end=100, value=0, step=0.1, width=100)

metrics_select = pn.widgets.Select(name='Metrics', options=['MSD', 'PL'], width=100)

# Define main function
@pn.depends(
  number_of_steps_input,
  trajectory_type,
  speed_input,
  start_x_pos_input,
  start_y_pos_input,
  cauchy_coefficient_input,
  alpha_input,
  beta_input,
  loc_input,
  metrics_select
)
def get_plots(
  number_of_steps_input, trajectory_type, speed_input, start_x_pos_input, start_y_pos_input, cauchy_coefficient_input,
  alpha_input, beta_input, loc_input, metrics_select
):
  
  changed_parameter = None

  # List of parameters to be saved in the same order as defined in @pn.depends
  parameters = [
      number_of_steps_input, trajectory_type, speed_input, start_x_pos_input, start_y_pos_input,
      cauchy_coefficient_input, alpha_input, beta_input, loc_input, metrics_select
  ]

  # Compare current values with saved values
  if not hasattr(get_plots, 'last_parameters'):
      get_plots.last_parameters = parameters
  else:
      for current_value, last_value in zip(parameters, get_plots.last_parameters):
          if current_value != last_value:
              changed_parameter = current_value
              break

      # Update saved parameters
      get_plots.last_parameters = parameters

  # Check for metric_select change
  if changed_parameter == None or changed_parameter not in ("MSD", "PL"):

    global trajectory_df

    if trajectory_type == 'BM':
      trajectory_df = bm_2d(
        n_steps=number_of_steps_input,
        speed=speed_input,
        s_pos=[start_x_pos_input, start_y_pos_input],
      )

    elif trajectory_type == 'CRW':
      trajectory_df = crw_2d(
        n_steps=number_of_steps_input,
        speed=speed_input,
        s_pos=[start_x_pos_input, start_y_pos_input],
        exponent=cauchy_coefficient_input
      )

    elif trajectory_type == 'LF':
      trajectory_df = lf_2d(
        n_steps=number_of_steps_input,
        speed=speed_input,
        s_pos=[start_x_pos_input, start_y_pos_input],
        CRW_exponent=cauchy_coefficient_input,
        alpha=alpha_input,
        beta=beta_input,
        loc=loc_input
      )
    else:
      trajectory_df = bm_2d(
        n_steps=number_of_steps_input,
        speed=speed_input,
        s_pos=[start_x_pos_input, start_y_pos_input],
      )
    
    # Plot trajectory
    global fig_trajectory
    fig_trajectory = go.Figure()

    time_z = np.linspace(0, 1, number_of_steps_input)
    fig_trajectory.add_trace(
        go.Scatter3d(
            x=trajectory_df.x_pos,
            y=trajectory_df.y_pos,
            z=time_z,
            name=f'steps: {number_of_steps_input}',
            showlegend=True,
            marker=dict(
                size=1,
            ),
            line=dict(
                width=2,
            )
        )
    )

    fig_trajectory.update_layout(title=f'{trajectory_type} Trajectory', showlegend=True, width=500, height=500)

  # Plot Metric
  if metrics_select == 'MSD':
    metric_pd = get_mean_squared_displacement(trajectory_df)
  elif metrics_select == 'PL':
    metric_pd = get_path_length(trajectory_df)
  else:
    metric_pd = get_mean_squared_displacement(trajectory_df)

  resolution = number_of_steps_input
  aux_domain = np.linspace(0, number_of_steps_input, resolution)
  fig_metric = go.Figure()
  fig_metric.add_trace(
      go.Scatter(
          x=aux_domain,
          y=metric_pd.data,
          name=f'{metrics_select}',
          showlegend=True
      )
  )

  fig_metric.update_layout(title=f'{trajectory_type} {metrics_select} Metric', showlegend=True, width=500, height=500)

  return pn.Row(fig_trajectory, fig_metric)


@pn.depends(trajectory_type)
def define_parameters(trajectory_type):

  if trajectory_type == 'BM':
    column = pn.Column(
      pn.Row(number_of_steps_input, speed_input),
      pn.Row(start_x_pos_input, start_y_pos_input),
    )

  elif trajectory_type == 'CRW':
    column = pn.Column(
      pn.Row(number_of_steps_input, speed_input),
      pn.Row(start_x_pos_input, start_y_pos_input),
      cauchy_coefficient_input,
    )

  elif trajectory_type == 'LF':
    column = pn.Column(
      pn.Row(number_of_steps_input, speed_input),
      pn.Row(start_x_pos_input, start_y_pos_input),
      pn.Row(cauchy_coefficient_input, loc_input),
      pn.Row(alpha_input, beta_input)
    )

  else:
    column = pn.Column(
      pn.Row(number_of_steps_input, speed_input),
      pn.Row(start_x_pos_input, start_y_pos_input)
    )
  
  return column

# Group parameters
group1 = pn.Column(
    pn.Row(trajectory_type),
    define_parameters,
    pn.Row(metrics_select)
)


# Create Column with both plots
plot_panel = pn.Column(get_plots)

# Create final layout
layout = pn.Column(
    pn.Row(group1, plot_panel)
)

# Show
layout.show()
