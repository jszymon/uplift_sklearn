"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

from usklearn.datasets import fetch_Hillstrom

D = fetch_Hillstrom()
print(D[:3])
