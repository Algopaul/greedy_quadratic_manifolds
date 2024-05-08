## Greedy quadratic manifolds

Code for the computation of quadratic manifolds using a greedy column selection as proposed in
```bibtex
@Misc{SchwerdtnerP2024Greedy,
    title	= {Greedy construction of quadratic manifolds for nonlinear dimensionality reduction and nonlinear model reduction},
    author	= {Paul Schwerdtner and Benjamin Peherstorfer},
    doi		= {10.48550/arXiv.2403.06732},
    year	= {2024},
    archiveprefix={arXiv},
    primaryclass= {math.NA}
}
```

### Installation
Requires python3.11. Create new virtual environment and install requirements:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Usage
The greedy column selection is demonstrated in the notebook `greedy_quad_manifolds.ipynb`; alternatively, the code can be run as follows from the command line:
```bash
python greedyqm/quadmani.py
```
This computes a quadratic manifold of dimension 20 for the linear advection dataset. Parameters are passed as flags, e.g.
```bash
python greedyqm/quadmani.py --reduced_dimension=10 --n_vectors_to_check=100 --reg_magnitude=1e-8
```
This computes a quadratic manifold of dimension 10 for the linear advection dataset with a regularization magnitude of 1e-8 and chooses the next column from the next 100 singular vectors.

If you want to use the greedy column selection in your own code, you can use the function `quadmani_greedy` from `greedyqm/quadmani.py`. The function signature is
```python
quadmani_greedy(data_points, reduced_dimension, n_vectors_to_check, reg_magnitude, feature_map)
```
Here `data_points` is a `jax.numpy` array of shape `(n_state, n_samples)`, `reduced_dimension` is the desired dimension of the quadratic manifold, `n_vectors_to_check` is the number of singular vectors to consider for the greedy column selection, and `reg_magnitude` is the regularization magnitude. The `feature_map` is a nonlinear operation on the reduced data points, which is set to the Kronecker product by default, but other nonlinear functions are possible.
