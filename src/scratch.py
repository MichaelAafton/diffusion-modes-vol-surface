from src.simulate import generate_synthetic_dataset, StochasticHeatEquation, SPDEConfig
from src.pca import run_pca
import numpy as np

cfg = SPDEConfig(D=0.05, n_z=50, seed=42)
spde = StochasticHeatEquation(cfg)
surface = spde.simulate(20000)                 # long run → small statistical error
pca = run_pca(np.diff(surface, axis=0), n_components=10)

print("empirical :", np.round(pca.explained_variance_ratio, 4))
print("theory    :", np.round(spde.theoretical_increment_eigenvalues(10), 4))