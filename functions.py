import numpy as np
import pandas as pd

from scipy.stats import wrapcauchy
from scipy.stats import levy_stable
from scipy.spatial import distance

from classes import Vec2d

def turning_angle(pos_a, pos_b, pos_c):
    """
    Arguments:
        pos_a: First position coordinates
        pos_b: Second position coordinates
        pos_c: Third position coordinates
    Returns:
        theta: Turning angle
    """
    vector_ab = [pos_b, pos_a]
    norm_ab = np.linalg.norm(vector_ab) # Norma / Magnitud de vector AB

    vector_bc = [pos_c, pos_b]
    norm_bc = np.linalg.norm(vector_bc) # Norma / Magnitud de vector AB

    dot_p = np.dot(vector_ab, vector_bc) # Producto punto

    # Nota: Evitar division por cero con np.finfo(float).eps
    cos_theta = (dot_p) / np.finfo((norm_ab * norm_bc) ).eps

    # Angle orientation
    cross_p = np.cross(vector_ab, vector_bc)

    orient = np.sign(cross_p)
    if orient == 0:
        orient = 1

    theta = np.arccos(np.around(cos_theta,4)) * orient
    return theta


# Brwonian Motion Trajectory
def bm_2d(n_steps=1000, speed=5, s_pos=[0,0]):
  """
  Arguments:
    n_steps:
    speed:
    s_pos:
  Returns:
    BM_2d_df:
  """

  # Data Frame
  BM_2d_df = pd.DataFrame(columns=['x_pos', 'y_pos'])
  temp_df = pd.DataFrame([{
      'x_pos': s_pos[0],
      'y_pos': s_pos[1]
  }])

  BM_2d_df = pd.concat([BM_2d_df, temp_df], ignore_index=True)

  # Trajectory
  velocity = Vec2d(speed, 0)
  for i in range(n_steps - 1):

    # Select turn angle
    turn_angle = np.random.uniform(low=-np.pi, high=np.pi)
    velocity = velocity.rotated(turn_angle)

    # Update location
    temp_df = pd.DataFrame([{
      'x_pos': BM_2d_df.x_pos[i] + velocity.x,
      'y_pos': BM_2d_df.y_pos[i] + velocity.y
    }])
    BM_2d_df = pd.concat([BM_2d_df, temp_df], ignore_index=True)

  return BM_2d_df

# Correlated Random Walk Trajectory
def crw_2d(n_steps=1000, speed=5, s_pos=[0,0], exponent=0.6):
  """
  Arguments:
    n_steps:
    speed:
    s_pos:
    exponent:
  Returns:
    CRW_2d_df:
  """

  # Data Frame
  CRW_2d_df = pd.DataFrame(columns=['x_pos', 'y_pos'])
  temp_df = pd.DataFrame([{
      'x_pos': s_pos[0],
      'y_pos': s_pos[1]
  }])

  CRW_2d_df = pd.concat([CRW_2d_df, temp_df], ignore_index=True)

  # Trajectory
  velocity = Vec2d(speed, 0)
  for i in range(1, n_steps):

    # Select turn angle
    turn_angle = wrapcauchy.rvs(exponent)
    velocity = velocity.rotated(turn_angle)

    # Update location
    temp_df = pd.DataFrame([{
      'x_pos': CRW_2d_df.x_pos[i - 1] + velocity.x,
      'y_pos': CRW_2d_df.y_pos[i - 1] + velocity.y
    }])
    CRW_2d_df = pd.concat([CRW_2d_df, temp_df], ignore_index=True)

  return CRW_2d_df

# Levy Flight Trajectory
def lf_2d(n_steps=1000, speed=6, s_pos=[0,0], alpha=0.5, beta=1.0, loc=1.0, CRW_exponent=0.5):
  """
  Arguments:
    n_steps:
    speed:
    s_pos:
    alpha:
    beta:
    loc:
    CRW_exponent:
  Returns:
    LF_2d_df:
  """

  velocity = Vec2d(speed, 0)

  # Data Frame
  LF_2d_df = pd.DataFrame(columns=['x_pos', 'y_pos'])
  temp_df = pd.DataFrame([{
      'x_pos': s_pos[0],
      'y_pos': s_pos[1]
  }])
  LF_2d_df = pd.concat([LF_2d_df, temp_df], ignore_index=True)

  # Trajectory
  for i in range(0, n_steps):

    # Turn angle
    turn_angle = wrapcauchy.rvs(c=CRW_exponent)

    # Step lenght
    step_length = levy_stable.rvs(alpha=alpha, beta=beta, loc=loc)

    # Rotate
    velocity = velocity.rotated(turn_angle)

    # Update location
    temp_df = pd.DataFrame([{
        'x_pos': LF_2d_df.x_pos[i] + (velocity.x * step_length),
        'y_pos': LF_2d_df.y_pos[i] + (velocity.y * step_length)
    }])
    LF_2d_df = pd.concat([LF_2d_df, temp_df], ignore_index=True)

  return LF_2d_df

# Define your function to compute path length for given trajectory
def get_path_length(trajectory):

    distance_df = np.array([distance.euclidean(trajectory.iloc[i-1], trajectory.iloc[i]) for i in range(1, trajectory.shape[0])])
    path_length = np.array([sum(distance_df[:i+1]) for i in range(len(distance_df))])
    path_length_df = pd.DataFrame(path_length, columns=['data'])
    return path_length_df

# Define your function to compute Mean Squared Displacement for given trajectory
def get_mean_squared_displacement(trajectory):

    N = len(trajectory)
    msd_list = []

    for n in range(1, N):

      squared_distances = []
      for i in range(N - n):

        distance_value = distance.euclidean(trajectory.iloc[i], trajectory.iloc[i + n])
        squared_distances.append(pow(distance_value, 2))

      msd_list.append(np.sum(squared_distances) / (N - n))

    msd_list_df = pd.DataFrame(msd_list, columns=['data'])
    return msd_list_df


# Define your function to compute Turning Angles for given trajectory
def get_turning_angles(trajectory):
    """
    Get Turnin angles for given trajectory

    Arguments:
        trajectory: Full trajectory data frame
    Returns:
        turning_angles_df: Turning angles from trajectories
    """

    turning_angles = []
    for i in range(1, len(trajectory) - 1):

      # Positions
      pos_a = trajectory.iloc[i - 1]  # Row i - 1
      pos_b = trajectory.iloc[i]      # Row i
      pos_c = trajectory.iloc[i + 1]  # Row i + 1

      # Vectors
      vector_ab = pos_b - pos_a
      vector_bc = pos_c - pos_b

      # Norm / AB Vector Magnitude
      norm_ab = np.linalg.norm(vector_ab)
      norm_bc = np.linalg.norm(vector_bc)

      dot_p = np.dot(vector_ab, vector_bc) # Dot product
      cos_theta = dot_p / (norm_ab * norm_bc)

      # Angle orientation
      cross_p = np.cross(vector_ab, vector_bc)

      orientation = np.sign(cross_p)

      if orientation == 0:
          orientation = 1

      # Get angle in radians
      theta = np.arccos(np.around(cos_theta, 4)) * orientation
      turning_angles.append(theta)

    turning_angles_df = pd.DataFrame(turning_angles, columns=['angles'])
    return turning_angles_df

# Define your function to compute Step lengths for given trajectory
def get_step_lengths(trajectory):
    """
    Get step lengths given trajectory

    Arguments:
        trajectory: DataFrame de la trayectoria con las coordenadas de cada punto.

    Return:
        step_lengths_df: DataFrame con las longitudes de los pasos.
    """

    turning_angles = get_turning_angles(trajectory)

    step_lengths = []
    steps = 1
    for angle in turning_angles.angles:

      if abs(np.around(angle, 4)) < 0.2:

        step_lengths.append(steps)
        steps = 0

      steps += 1

    step_lengths_df = pd.DataFrame(step_lengths, columns=['steps'])
    return step_lengths_df