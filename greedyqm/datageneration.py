"""Generates a pulse that advects with a given speed."""
import jax
import jax.numpy as jnp
import numpy as np
from absl import flags, app

PULSE_WIDTH = flags.DEFINE_float('pulse_width', 2.0e-4, 'Width of the pulse')
PULSE_SHIFT = flags.DEFINE_float('pulse_shift', 1.0e-1, 'Shift of the pulse')
SPEED = flags.DEFINE_float('speed', 10.0, 'Speed of the pulse')
FINAL_TIME = flags.DEFINE_float('final_time', 0.1,
                                'Final time of the simulation')
N_TIME_SAMPLES = flags.DEFINE_integer('n_time_samples', 2000,
                                      'Number of time samples')
N_SPACE_SAMPLES = flags.DEFINE_integer('n_space_samples', 2**12,
                                       'Number of space samples')


def gaussian_pulse(x, pulse_width=None, pulse_shift=None):
  if pulse_width is None:
    pulse_width = PULSE_WIDTH.value
  if pulse_shift is None:
    pulse_shift = PULSE_SHIFT.value
  return 1 / jnp.sqrt(pulse_width * jnp.pi) * jnp.exp(-(
      (x - pulse_shift))**2 / pulse_width)


def generate_advecting_pulse(pulse_width=None,
                             pulse_shift=None,
                             speed=None,
                             final_time=None,
                             n_time_samples=None,
                             n_space_samples=None):
  if pulse_width is None:
    pulse_width = PULSE_WIDTH.value
  if pulse_shift is None:
    pulse_shift = PULSE_SHIFT.value
  if speed is None:
    speed = SPEED.value
  if final_time is None:
    final_time = FINAL_TIME.value
  if n_time_samples is None:
    n_time_samples = N_TIME_SAMPLES.value
  if n_space_samples is None:
    n_space_samples = N_SPACE_SAMPLES.value

  x = jnp.linspace(0, 1, n_space_samples)
  t = jnp.linspace(0, final_time, n_time_samples)

  def f(x, ti):
    u = gaussian_pulse(x - speed * ti, pulse_width, pulse_shift)
    return x, u

  _, data_points = jax.lax.scan(f, x, t)

  return data_points.T


def main(_):
  data_points = generate_advecting_pulse()
  np.save('advecting_pulse.npy', data_points)


if __name__ == '__main__':
  app.run(main)
