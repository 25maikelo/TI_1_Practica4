import panel as pn
import numpy as np
import plotly.graph_objects as go
from functions import bm_2d, crw_2d, lf_2d, get_mean_squared_displacement, get_step_lengths, get_turning_angles


trajectory = []

# Crear los widgets
# TODO: Arreglar punto flotante
# TODO: Decidir Metrica conforme el tipo de trayectoria
# TODO: Verificar métrica
# TODO: Acomodar trayectoria inicial vacía: no sé si debería estar como una variable externa fuera de las funciones

trajectory_type = pn.widgets.RadioButtonGroup(name='Botones', options=['BM', 'CRW', 'LF'])
number_of_steps_input = pn.widgets.IntInput(name='Number of steps', start=1, end=10000, value=100)
speed_input = pn.widgets.FloatInput(name='Speed', start=1, end=100, value=6)
start_x_pos_input = pn.widgets.FloatInput(name='Start X pos', start=0, end=100, value=0)
start_y_pos_input = pn.widgets.FloatInput(name='Start Y pos', start=0, end=100, value=0)

cauchy_coefficient_input = pn.widgets.FloatInput(name='Cauchy Coefficient', start=0.1, end=0.9, value=0.1, step=0.1)

alpha_input = pn.widgets.FloatInput(name='Alpha', start=0.1, end=3, value=1, step=0.1)
beta_input = pn.widgets.FloatInput(name='Beta', start=0.1, end=np.pi, value=1, step=0.1)
loc_input = pn.widgets.FloatInput(name='Loc', start=0, end=100, value=0, step=0.1)

metrics_select = pn.widgets.Select(name='Metrics', options=['MSD', 'PL', 'TA'])


@pn.depends(metrics_select)
def plot_metric(metrics_select):
  if metrics_select == 'MSD':
    metric_pd = get_mean_squared_displacement(trajectory)
  elif metrics_select == 'PL':
    metric_pd = get_step_lengths(trajectory)
  elif metrics_select == 'TA':
    metric_pd = get_turning_angles(trajectory)
  else:
    metric_pd = get_turning_angles(trajectory)

  x_values = np.linspace(0, 100)
  fig_traj_rw = go.Figure()
  fig_traj_rw.add_trace(
      go.Scatter(
          x=x_values,
          y=metric_pd,
          name=f'Metric: {metrics_select}',
          showlegend=True
      )
  )

  return fig_traj_rw

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
  loc_input
)
def plot_trajectory(
  number_of_steps_input, trajectory_type, speed_input, start_x_pos_input, start_y_pos_input, cauchy_coefficient_input,
  alpha_input, beta_input, loc_input
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

  trajectory = trajectory_df
  
  fig_trajectory = go.Figure()
  fig_trajectory.add_trace(
      go.Scatter(
          x=trajectory_df.x_pos,
          y=trajectory_df.y_pos,
          name=f'steps: {number_of_steps_input}',
          showlegend=True
      )
  )

  fig_trajectory.update_layout(title=f'{trajectory_type} Trajectory', showlegend=True)
  return fig_trajectory


@pn.depends(trajectory_type)
def define_parameters(trajectory_type):

  if trajectory_type == 'BM':
    column = pn.Column(
      number_of_steps_input,
      start_x_pos_input,
      start_y_pos_input,
      speed_input,
    )

  elif trajectory_type == 'CRW':
    column = pn.Column(
      number_of_steps_input,
      start_x_pos_input,
      start_y_pos_input,
      speed_input,
      cauchy_coefficient_input,
    )

  elif trajectory_type == 'LF':
    column = pn.Column(
      number_of_steps_input,
      start_x_pos_input,
      start_y_pos_input,
      speed_input,
      cauchy_coefficient_input,
      alpha_input,
      beta_input,
      loc_input,
    )

  else:
    column = pn.Column(
      number_of_steps_input,
      start_x_pos_input,
      start_y_pos_input,
      speed_input,
    )
  
  return column

# Organizar los widgets en grupos
group1 = pn.Column(
    pn.Row(trajectory_type),
    define_parameters,
    pn.Row(metrics_select)
)

# Crear los paneles con las gráficas
plot_panel1 = pn.Column(plot_trajectory)
plot_panel2 = pn.Column(plot_metric)

# Crear el layout final
layout = pn.Column(
    pn.Row(group1, plot_panel1, plot_panel2)
)

# Mostrar el panel
layout.show()
