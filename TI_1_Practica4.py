import panel as pn
import panel.widgets as pnw
import plotly.graph_objects as go

from functions import bm_2d

pn.extension('plotly')


# Definir el widget para ingresar el número de pasos
n_steps = pnw.IntInput(name='Number of steps:', value=20, step=10, start=1, end=1000)

@pn.depends(n_steps)
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

column = pn.Column(n_steps, plot_traj)
column.show()