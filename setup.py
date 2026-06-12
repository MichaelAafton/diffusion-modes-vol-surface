from setuptools import setup, find_packages

setup(
    name="vol-surface-dynamics",
    version="0.1.0",
    description=(
        "Heat-equation dynamics in the volatility surface: testing a "
        "field-theory factor model against real options data."
    ),
    packages=find_packages(),
    python_requires=">=3.10",
)
