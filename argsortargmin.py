# -*- coding: utf-8 -*-
"""
Purpose: To show how to use argmin and argsort
on parallel arrays
"""

import numpy as np

# Parallel arrays: The indices must be in correspondence
names = ["David", "Kat", "Max", "Matt", "Chris", "Sam", "Theo"]
ages = np.array([23, 20, 21, 20, 31, 20, 20])
#index = np.argmax(ages)
#print(names[index])
print(np.argsort(-ages))
for index in np.argsort(-ages):
    print(names[index])