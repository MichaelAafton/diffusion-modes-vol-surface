from setuptools import setup, find_packages

setup(
    name="vol-surface-dynamics",
    version="1.0.0",
    description=(
        "Spectral structure of the 30-day SPX implied-volatility smile "
        "in vendor-smoothed surfaces."
    ),
    packages=find_packages(),
    python_requires=">=3.10",
)
