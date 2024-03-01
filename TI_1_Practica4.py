import panel as pn
import numpy as np
import plotly.graph_objects as go
from functions import bm_2d



# Crear los widgets
button_group1 = pn.widgets.RadioButtonGroup(name='Botones', options=['BM', 'CRW', 'LF'])
number_of_steps_input = pn.widgets.IntInput(name='Number of steps', start=1, end=10000, value=1)
start_x_pos_input = pn.widgets.FloatInput(name='Start X pos', start=0, end=100, value=0)
start_y_pos_input = pn.widgets.FloatInput(name='Start Y pos', start=0, end=100, value=0)
cauchy_coefficient_input = pn.widgets.FloatInput(name='Cauchy Coefficient', start=0.1, end=0.9, value=0.1, step=0.1)
metrics_select = pn.widgets.Select(name='Metrics', options=['MSD', 'PL'])

# Definir la función para graficar
@pn.depends(number_of_steps_input)
def plot_traj(n_steps):

  random_walker_df = bm_2d(n_steps)
  fig_traj_rw = go.Figure()
  fig_traj_rw.add_trace(
      go.Scatter(
          x=random_walker_df.x_pos,
          y=random_walker_df.y_pos,
          name=f'steps: {n_steps}',
          showlegend=True
      )
  )

  return fig_traj_rw

# Organizar los widgets en grupos
group1 = pn.Column(
    pn.Row(button_group1),
    pn.Column(number_of_steps_input, start_x_pos_input, start_y_pos_input, cauchy_coefficient_input),
    pn.Row(metrics_select)
)

# Crear los paneles con las gráficas
plot_panel1 = pn.Column(plot_traj)
plot_panel2 = pn.Column(plot_traj)

# Crear el layout final
layout = pn.Column(
    pn.Row(group1, plot_panel1, plot_panel2)
)

# Mostrar el panel
layout.show()
