# -*- coding: utf-8 -*-
"""
Purpose: Show how to make parallel arrays of the
keys/values in a dictionary and to use armax
on the values to index into the keys
"""
import numpy as np

fin = open("Iliad.txt", "r")
iliad_text = fin.read()
fin.close()

iliad_text = iliad_text.lower()
counts_dict = {} # key is a word, value is its counts
for word in iliad_text.split():
    if word in counts_dict:
        counts_dict[word] = counts_dict[word] + 1
    else:
        counts_dict[word] = 1
words = list(counts_dict.keys())
counts = list(counts_dict.values())
index = np.argmax(counts)
print(index)
print(words[index])
print(counts[index])