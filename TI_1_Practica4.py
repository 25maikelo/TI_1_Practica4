import panel as pn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functions import bm_2d, crw_2d, lf_2d, get_mean_squared_displacement, get_path_length


trajectory = pd.DataFrame()

trajectory_type = pn.widgets.RadioButtonGroup(name='Botones', options=['BM', 'CRW', 'LF'])
number_of_steps_input = pn.widgets.IntInput(name='Number of steps', start=1, end=10000, value=100, width=100)
speed_input = pn.widgets.FloatInput(name='Speed', start=1, end=100, value=6, width=100)
start_x_pos_input = pn.widgets.FloatInput(name='Start X pos', start=0, end=100, value=0, width=100)
start_y_pos_input = pn.widgets.FloatInput(name='Start Y pos', start=0, end=100, value=0, width=100)

cauchy_coefficient_input = pn.widgets.FloatInput(name='Cauchy Coefficient', start=0.1, end=0.9, value=0.1, step=0.1, width=100)

alpha_input = pn.widgets.FloatInput(name='Alpha', start=0.1, end=3, value=1, step=0.1, width=100)
beta_input = pn.widgets.FloatInput(name='Beta', start=0.1, end=np.pi, value=1, step=0.1, width=100)
loc_input = pn.widgets.FloatInput(name='Loc', start=0, end=100, value=0, step=0.1, width=100)

metrics_select = pn.widgets.Select(name='Metrics', options=['MSD', 'PL'], width=100)

# Definir la función para graficar
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
def plot_trajectory(
  number_of_steps_input, trajectory_type, speed_input, start_x_pos_input, start_y_pos_input, cauchy_coefficient_input,
  alpha_input, beta_input, loc_input, metrics_select
):

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
  
  fig_trajectory = go.Figure()
  fig_trajectory.add_trace(
      go.Scatter(
          x=trajectory_df.x_pos,
          y=trajectory_df.y_pos,
          name=f'steps: {number_of_steps_input}',
          showlegend=True
      )
  )

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

  fig_trajectory.update_layout(title=f'{trajectory_type} Trajectory', showlegend=True, width=500, height=500)

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

# Organizar los widgets en grupos
group1 = pn.Column(
    pn.Row(trajectory_type),
    define_parameters,
    pn.Row(metrics_select)
)


# Crear los paneles con las gráficas
plot_panel = pn.Column(plot_trajectory)

# Crear el layout final
layout = pn.Column(
    pn.Row(group1, plot_panel)
)

# Mostrar el panel
layout.show()
