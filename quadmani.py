import jax
import jax.numpy as jnp
from jax.lax import fori_loop

jax.config.update('jax_enable_x64', True)

def default_feature_map(reduced_data_points):
  r = reduced_data_points.shape[0]
  return jnp.concatenate(
    [reduced_data_points[i] * reduced_data_points[:i + 1] for i in range(r)],
    axis=0
  )


def quadmani_greedy(data_points, r=20, n_vectors_to_check=200, feature_map=default_feature_map):
  shift_value = jnp.mean(data_points, axis=1)
  Phi, Sigma, PsiT = jnp.linalg.svd(shift_data(data_points, -shift_value))
  idx_in = jnp.arange(0, 0, 1)
  idx_out = jnp.arange(0, len(Sigma), 1)
  idx_in, idx_out = greedy_step_fast(
      idx_in,
      idx_out,
      Sigma,
      PsiT,
      imax=n_vectors_to_check,
      nonlinear_map=feature_map,
  )
  for _ in range(r - 1):
    idx_in, idx_out = greedy_step_fast(
        idx_in,
        idx_out,
        Sigma,
        PsiT,
        imax=n_vectors_to_check,
        nonlinear_map=feature_map,
    )
  V = Phi[:, idx_in]
  SIGMA1, SIGMA2 = Sigma[idx_in], Sigma[idx_out]
  V1T, V2T = PsiT[idx_in, :], PsiT[idx_out, :]
  embedded_snapshots = jnp.diag(SIGMA1) @ V1T
  V2S2 = V2T.T * SIGMA2
  H = feature_map(embedded_snapshots)
  W, _ = lstsq_l2(H.T, V2S2)
  return V, Phi[:, idx_out] @ W.T, shift_value


def lstsq_l2(A, B, reg_magnitude=1e-6):
  PHI, SIGMA, PSI_T = jnp.linalg.svd(A, full_matrices=False)
  sinv = SIGMA / (SIGMA**2 + reg_magnitude**2)
  x = PSI_T.T * sinv @ (PHI.T @ B)
  B_estimate = A @ x
  resid = jnp.linalg.norm(B - B_estimate)
  return x, resid


def greedy_step_fast(
    idx_in_pre, idx_out_pre, SIGMA, VT, imax, nonlinear_map
):
  def body_fun(i, errors):
    idx_in = jnp.hstack((idx_in_pre, idx_out_pre[i]))
    idx_out = jnp.delete(idx_consider, i, assume_unique_indices=True)
    errors = errors.at[i].set(
        compute_error_fast(idx_in, idx_out, SIGMA, VT, nonlinear_map)
    )
    return errors

  n_consider = jnp.minimum(imax, len(idx_out_pre))
  errors = jnp.zeros(shape=(n_consider,))
  idx_consider = idx_out_pre[:n_consider]
  errors = fori_loop(0, len(errors), body_fun, errors)
  idx = jnp.argmin(errors)
  idx_in_next = jnp.hstack((idx_in_pre, idx_out_pre[idx]))
  idx_out_next = jnp.delete(idx_out_pre, idx)
  return idx_in_next, idx_out_next


def compute_error_fast(idx_in, idx_out, SIGMA, VT, feature_map):
  SIGMA1, SIGMA2 = SIGMA[idx_in], SIGMA[idx_out]
  V1T, V2T = VT[idx_in, :], VT[idx_out, :]
  embedded_snapshots = jnp.diag(SIGMA1) @ V1T
  V2S2 = V2T.T * SIGMA2
  W = feature_map(embedded_snapshots)
  _, residuals = lstsq_l2(W.T, V2S2)
  return residuals


def shift_data(data_points, shift):
  return (data_points.T - shift.T).T


def linear_reduce(V, data_points, shift_value):
  return V.T @ shift_data(data_points, -shift_value)


def lift_quadratic(V, W, shift_value, reduced_data_points, feature_map=default_feature_map):
  linear_part = V @ reduced_data_points
  quadratic_part = W @ feature_map(reduced_data_points)
  return shift_data(linear_part + quadratic_part, shift_value)


if __name__ == "__main__":
  PULSE_WIDTH = 2.0e-4
  PULSE_SHIFT = 1.0e-1
  SPEED = 10.0
  FINAL_TIME = 0.1
  N_TIME_SAMPLES = 2000
  N_SPACE_SAMPLES = 2**12

  def gaussian_pulse(x):
    return 1 / jnp.sqrt(PULSE_WIDTH*jnp.pi) * jnp.exp(-((x-PULSE_SHIFT))**2 / PULSE_WIDTH)

  def generate_data():
    t = jnp.linspace(0, FINAL_TIME, N_TIME_SAMPLES)

    def f(x, ti):
      u = gaussian_pulse(x-SPEED * ti)
      return x, u
    
    _, data_points = jax.lax.scan(f, x, t)
    return data_points.T

  x = jnp.linspace(0, 1, N_SPACE_SAMPLES)
  data_points = generate_data()
  train_points = data_points[:, ::2]
  test_points = data_points[:, 1::2]
  V, W, shift_value = quadmani_greedy(data_points)
  reduced_points = linear_reduce(V, test_points, shift_value)
  reconstructed = lift_quadratic(V, W, shift_value, reduced_points)
  assert jnp.linalg.norm(reconstructed-test_points)/jnp.linalg.norm(test_points) < 5e-7

