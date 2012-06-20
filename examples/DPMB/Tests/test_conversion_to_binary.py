#!python

import os
import cPickle
filename = os.path.join("Data","CIFAR10-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)

print cifar["codes"].shape

data = []
for row in cifar["codes"]:
  binary_rep = []
  for number in row:
    raw_string_rep = bin(number)[2:]
    string_rep = "0" * (32-len(raw_string_rep)) + raw_string_rep
    for value in string_rep:
      if value == "1":
        binary_rep.append(True)
      else:
        binary_rep.append(False)
  data.append(binary_rep)
