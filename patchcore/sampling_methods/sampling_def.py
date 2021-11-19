from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import pairwise_distances

import abc
import numpy as np

"""Abstract class for sampling methods.
Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None

"""Returns points that minimizes the maximum distance of any point to a center.
Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).
Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""