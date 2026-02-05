name: chemml
channels:
  - conda-forge
  - defaults

dependencies:
Python
  - python=3.11

Core numerical & data
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.10

Visualization
  - matplotlib>=3.7

Machine learning
  - scikit-learn>=1.3

Chemistry
  - rdkit>=2024

Tree / boosting models
  - xgboost>=2.0
  - lightgbm>=4.0

Deep learning
  - pytorch>=2.0

Graph neural networks
  - torch-geometric>=2.5

Jupyter
  - ipykernel
  - jupyter

Pip-only packages
  - pip
  - pip:
      - transformers>=4.40
      - datasets
      - accelerate
      - evaluate