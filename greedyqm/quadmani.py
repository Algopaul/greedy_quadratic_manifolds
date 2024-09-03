"""Quadratic manifold computation using greedy algorithm."""
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from greedyqm.datageneration import generate_advecting_pulse
from absl import app
from absl import flags
from absl import logging
from collections import namedtuple

jax.config.update('jax_enable_x64', True)

REDUCED_DIMENSION = flags.DEFINE_integer('reduced_dimension', 20,
                                         'Reduced dimension of the data')
N_VECTORS_TO_CHECK = flags.DEFINE_integer(
    'n_vectors_to_check', 200,
    'Number of vectors to check in the greedy algorithm')

REG_MAGNITUDE = flags.DEFINE_float(
    'reg_magnitude', 1e-6, 'Regularization magnitude for least squares problem')

IDX_IN = flags.DEFINE_multi_integer(
    'idx_in', [], 'Indices of the columns to be included in the reduced basis')

ShiftedSVD = namedtuple('ShiftedSVD', ['U', 'S', 'VT', 'shift'])


def default_feature_map(reduced_data_points):
  r = reduced_data_points.shape[0]
  return jnp.concatenate(
      [reduced_data_points[i] * reduced_data_points[:i + 1] for i in range(r)],
      axis=0)


def quadmani_greedy(
    data_points,
    r=None,
    n_vectors_to_check=None,
    reg_magnitude=None,
    idx_in=None,
    feature_map=default_feature_map,
):
  if r is None:
    r = REDUCED_DIMENSION.value
  if n_vectors_to_check is None:
    n_vectors_to_check = N_VECTORS_TO_CHECK.value
  if reg_magnitude is None:
    reg_magnitude = REG_MAGNITUDE.value
  if idx_in is None:
    idx_in = jnp.asarray(IDX_IN.value, dtype=jnp.int32)
  shift_value = jnp.mean(data_points, axis=1)
  phi, sigma, psit = jnp.linalg.svd(shift_data(data_points, shift_value))
  shifted_svd = ShiftedSVD(phi, sigma, psit, shift_value)
  return quadmani_greedy_from_svd(
      shifted_svd,
      r,
      n_vectors_to_check,
      reg_magnitude,
      idx_in,
      feature_map,
  )


def quadmani_greedy_from_svd(
    shifted_svd,
    r=None,
    n_vectors_to_check=None,
    reg_magnitude=None,
    idx_in=None,
    feature_map=default_feature_map,
):
  if r is None:
    r = REDUCED_DIMENSION.value
  if n_vectors_to_check is None:
    n_vectors_to_check = N_VECTORS_TO_CHECK.value
  if reg_magnitude is None:
    reg_magnitude = REG_MAGNITUDE.value
  if idx_in is None:
    idx_in = jnp.asarray(IDX_IN.value, dtype=jnp.int32)
  phi, sigma, psit, shift_value = shifted_svd
  print(idx_in)
  idx_out = jnp.setdiff1d(
      jnp.arange(0, len(sigma), 1), idx_in, assume_unique=True)
  i = 1
  while len(idx_in) < r:
    idx_in, idx_out = greedy_step_fast(
        idx_in,
        idx_out,
        sigma,
        psit,
        imax=n_vectors_to_check,
        nonlinear_map=feature_map,
        reg_magnitude=reg_magnitude,
    )
    logging.info('Greedy step %i. Selected columns %s', i, idx_in)
    i += 1
  V = phi[:, idx_in]
  sigma_1, sigma_2 = sigma[idx_in], sigma[idx_out]
  V1T, V2T = psit[idx_in, :], psit[idx_out, :]
  embedded_snapshots = jnp.diag(sigma_1) @ V1T
  V2S2 = V2T.T * sigma_2
  H = feature_map(embedded_snapshots)
  W, _ = lstsq_l2(H.T, V2S2, reg_magnitude=reg_magnitude)
  return V, phi[:, idx_out] @ W.T, shift_value


def lstsq_l2(A, B, reg_magnitude=None):
  if reg_magnitude is None:
    reg_magnitude = REG_MAGNITUDE.value
  phi, sigma, psi_t = jnp.linalg.svd(A, full_matrices=False)
  sinv = sigma / (sigma**2 + reg_magnitude**2)
  x = psi_t.T * sinv @ (phi.T @ B)
  B_estimate = A @ x
  resid = jnp.linalg.norm(B - B_estimate)
  return x, resid


def greedy_step_fast(idx_in_pre, idx_out_pre, sigma, VT, imax, nonlinear_map,
                     reg_magnitude):

  def body_fun(i, errors):
    idx_in = jnp.hstack((idx_in_pre, idx_out_pre[i]))
    idx_out = jnp.delete(idx_consider, i, assume_unique_indices=True)
    errors = errors.at[i].set(
        compute_error_fast(idx_in, idx_out, sigma, VT, nonlinear_map,
                           reg_magnitude))
    return errors

  n_consider = jnp.minimum(imax, len(idx_out_pre))
  errors = jnp.zeros(shape=(n_consider,))
  idx_consider = idx_out_pre[:n_consider]
  errors = fori_loop(0, len(errors), body_fun, errors)
  idx = jnp.argmin(errors)
  idx_in_next = jnp.hstack((idx_in_pre, idx_out_pre[idx]))
  idx_out_next = jnp.delete(idx_out_pre, idx)
  return idx_in_next, idx_out_next


def compute_error_fast(idx_in, idx_out, sigma, VT, feature_map, reg_magnitude):
  sigma_1, sigma_2 = sigma[idx_in], sigma[idx_out]
  V1T, V2T = VT[idx_in, :], VT[idx_out, :]
  embedded_snapshots = jnp.diag(sigma_1) @ V1T
  V2S2 = V2T.T * sigma_2
  W = feature_map(embedded_snapshots)
  _, residuals = lstsq_l2(W.T, V2S2, reg_magnitude)
  return residuals


def shift_data(data_points, shift):
  return (data_points.T - shift.T).T


def linear_reduce(V, data_points, shift_value):
  return V.T @ shift_data(data_points, shift_value)


def lift_quadratic(V,
                   W,
                   shift_value,
                   reduced_data_points,
                   feature_map=default_feature_map):
  linear_part = V @ reduced_data_points
  quadratic_part = W @ feature_map(reduced_data_points)
  return shift_data(linear_part + quadratic_part, -shift_value)


def main(_):
  logging.info('Generating traveling pulse data')
  data_points = generate_advecting_pulse()
  logging.info('Dividing train and test points')
  train_points = data_points[:, ::2]
  test_points = data_points[:, 1::2]
  logging.info('Computing qudratic manifold')
  V, W, shift_value = quadmani_greedy(train_points)
  reduced_points = linear_reduce(V, test_points, shift_value)
  reconstructed = lift_quadratic(V, W, shift_value, reduced_points)
  logging.info('Checking the reconstruction error')
  rel_rec_error = jnp.linalg.norm(reconstructed -
                                  test_points) / jnp.linalg.norm(test_points)
  logging.info('Relative reconstruction error: %.3e', rel_rec_error)


if __name__ == '__main__':
  app.run(main)
