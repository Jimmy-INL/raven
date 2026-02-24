# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Repository of utils for non-dominated and Pareto frontier methods
  Created  Feb 18, 2020
  @authors: Diego Mandelli and Mohammad Abdo
"""
# External Imports
import numpy as np
import xarray as xr
# Internal Imports


def nonDominatedFrontier(data, returnMask, minMask=None, isFitness=False):
  """
    This method identifies the set of non-dominated points (nEfficientPoints).

    If returnMask=True, a True/False mask (nonDominatedFrontierMask) is returned.
    Non-dominated points pFront can be obtained as follows:
      mask = nonDominatedFrontier(data, True)
      pFront = data[np.array(mask)]

    If returnMask=False, an array of integer values containing the indexes of the non-dominated points is returned.
    Non-dominated points pFront can be obtained as follows:
      mask = nonDominatedFrontier(data, False)
      pFront = data[np.array(mask)]

    @ In, data, np.array, data matrix (nPoints, nCosts) containing the data points
    @ In, returnMask, bool, type of data to be returned: indices (False) or True/False mask (True)
    @ In, minMask, np.array, array (nCosts,) of boolean values: True (dimension needs to be minimized), False (dimension needs to be maximized)
    @ In, isFitness, boolean, if True, thus means the data is fitness values, otherwise, objective values (i.e., do not include penalties from constraint violation))
    @ Out, nonDominatedFrontierMask, np.array, data matrix (nPoints,), array of boolean values if returnMask=True
    @ Out, nonDominatedFrontier, np.array, data matrix (nEfficientPoints,), integer array of indexes if returnMask=False

    Reference: Adapted from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
  """

  if minMask is None:
    pass
  elif minMask is not None and len(minMask) != data.shape[1]:
    raise IOError("nonDominatedFrontier method: Data features do not match minMask dimensions: data has shape " + str(data.shape) + " while minMask has shape " + str(minMask.shape))
  elif not isFitness:
    for index, elem in enumerate(minMask):
      if not elem:
        data[:, index] = -1. * data[:, index]

  nPoints = data.shape[0]
  nonDominatedFrontier = np.arange(nPoints)
  nextPointIndex = 0

  while nextPointIndex < np.shape(data)[0]:
    if not isFitness:
      nondominatedPointMask = np.any(data < data[nextPointIndex], axis=1) | np.all(data == data[nextPointIndex], axis=1)

    else:
      nondominatedPointMask = np.any(data > data[nextPointIndex], axis=1) | np.all(data == data[nextPointIndex], axis=1)
    nonDominatedFrontier = nonDominatedFrontier[nondominatedPointMask]
    data = data[nondominatedPointMask]
    nextPointIndex = np.sum(nondominatedPointMask[:nextPointIndex]) + 1

  if returnMask:
    nonDominatedFrontierMask = np.zeros(nPoints, dtype=bool)
    nonDominatedFrontierMask[nonDominatedFrontier] = True
    return nonDominatedFrontierMask
  else:
    return nonDominatedFrontier


def rankNonDominatedFrontiers(data, isFitness=False):
  """
    This method ranks the non-dominated fronts by omitting the first front from the data
    and searching the remaining data for a new one recursively.
    @ In, data, np.array, data matrix (nPoints, nObjectives) containing the multi-objective
                          evaluations of each point/individual, element (i,j)
                          means jth objective/fitness function at the ith point/individual
    @ Out, nonDominatedRank, list, a list of length nPoints that has the ranking
                                  of the front passing through each point
  """
  nonDominatedRank = np.zeros(data.shape[0], dtype=int)
  mask = np.ones(data.shape[0], dtype=bool)
  rank = 0

  while np.any(mask):
    rank += 1
    # Get non-dominated points from remaining data
    if not isFitness:
      currentFront = nonDominatedFrontier(data[mask], False)
    else:
      currentFront = nonDominatedFrontier(data[mask], False, [False] * data.shape[1], isFitness=isFitness)
    # Convert indices back to original data space
    originalIndices = np.where(mask)[0][currentFront]
    # Assign rank
    nonDominatedRank[originalIndices] = rank
    # Update mask to remove current front
    mask[originalIndices] = False

  return nonDominatedRank.tolist()


def _applyMinMaxMask(objectives, minMask):
  """
    Convert objectives to a pure minimization form using the provided min/max mask.
    @ In, objectives, np.ndarray, shape (nPoints, nObjectives)
    @ In, minMask, list(bool) or None, True means minimize, False means maximize
    @ Out, transformed, np.ndarray, objectives transformed to minimize all dimensions
  """
  transformed = np.asarray(objectives, dtype=float)
  if minMask is None:
    return transformed
  if len(minMask) != transformed.shape[1]:
    raise IOError("minMask length does not match objective dimension")
  for idx, minimize in enumerate(minMask):
    if not minimize:
      transformed[:, idx] = -1.0 * transformed[:, idx]
  return transformed


def rankNonDominatedFrontiersObjectives(objectives, minMask=None):
  """
    Rank non-dominated fronts for objective values with mixed min/max directions.
    @ In, objectives, np.ndarray, shape (nPoints, nObjectives)
    @ In, minMask, list(bool) or None, True means minimize, False means maximize
    @ Out, ranks, list(int), non-dominated rank for each point
  """
  minimized = _applyMinMaxMask(objectives, minMask)
  return rankNonDominatedFrontiers(minimized, isFitness=False)


def rankNonDominatedFrontiersWithConstraints(objectives, constraints, minMask=None):
  """
    Rank non-dominated fronts using constraint-domination (Deb 2000).
    Feasible solutions dominate infeasible ones. Among infeasible, smaller total
    constraint violation is preferred. Among feasible, objective dominance applies.
    @ In, objectives, np.ndarray, shape (nPoints, nObjectives)
    @ In, constraints, np.ndarray or xr.DataArray, shape (nPoints, nConstraints)
    @ In, minMask, list(bool) or None, True means minimize, False means maximize
    @ Out, ranks, list(int), non-dominated rank for each point
  """
  obj = _applyMinMaxMask(objectives, minMask)
  if constraints is None:
    constraint_vals = np.zeros(obj.shape[0], dtype=float)
  else:
    g = np.asarray(constraints)
    if g.size == 0:
      constraint_vals = np.zeros(obj.shape[0], dtype=float)
    else:
      constraint_vals = np.sum(np.maximum(0.0, -g), axis=1)
  feasible = constraint_vals <= 0.0

  def dominates(i, j):
    if feasible[i] and not feasible[j]:
      return True
    if not feasible[i] and feasible[j]:
      return False
    if not feasible[i] and not feasible[j]:
      if constraint_vals[i] < constraint_vals[j]:
        return True
      return False
    return np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j])

  n_points = obj.shape[0]
  dominates_list = [set() for _ in range(n_points)]
  dominated_count = np.zeros(n_points, dtype=int)
  fronts = []

  for p in range(n_points):
    for q in range(n_points):
      if p == q:
        continue
      if dominates(p, q):
        dominates_list[p].add(q)
      elif dominates(q, p):
        dominated_count[p] += 1

  current_front = [idx for idx in range(n_points) if dominated_count[idx] == 0]
  rank = np.zeros(n_points, dtype=int)
  front_rank = 1
  while current_front:
    fronts.append(current_front)
    next_front = []
    for p in current_front:
      rank[p] = front_rank
      for q in dominates_list[p]:
        dominated_count[q] -= 1
        if dominated_count[q] == 0:
          next_front.append(q)
    front_rank += 1
    current_front = next_front

  return rank.tolist()


def crowdingDistance(rank, popSize, fitness):
  """
    Method designed to calculate the crowding distance for each front.

    FIXED: No longer assigns infinity to all points with boundary values.
    Only actual boundary points (first and last after sorting) get infinity.

    @ In, rank, np.array or xr.DataArray, array which contains the front ID for each element of the population
    @ In, popSize, int, size of population
    @ In, fitness, np.array, matrix contains fitness values for each element of the population
    @ Out, crowdDist, np.array, array of crowding distances
  """
  if isinstance(rank, xr.DataArray):
    rank = rank.data

  crowdDist = np.zeros(popSize)
  fronts = np.unique(rank)
  fronts = fronts[fronts != np.inf]

  # Keep track of which points are on each front
  frontIndices = {f: [] for f in fronts}
  for i, r in enumerate(rank):
    frontIndices[r].append(i)

  for f in fronts:
    front = frontIndices[f]  # Get indices of current front
    numObjectives = fitness.shape[1]
    numPoints = len(front)

    # Special case: fronts with ≤2 points
    if numPoints <= 2:
      crowdDist[front] = np.inf
      continue

    # For each objective, calculate crowding distance contribution
    for obj in range(numObjectives):
      # Sort points in current front by current objective
      sortedFront = [i for i in front]
      sortedIndices = np.argsort(fitness[sortedFront, obj], kind='stable')
      sortedFront = [sortedFront[i] for i in sortedIndices]

      # # To be removed
      # # ================================================================
      # # FIXED: Only set actual boundary points to infinity
      # # Do NOT set interior points with same values to infinity
      # # ================================================================
      # crowdDist[sortedFront[0]] = np.inf   # Minimum boundary
      # crowdDist[sortedFront[-1]] = np.inf  # Maximum boundary

      # # Skip normalization if all values are identical

      # # FIXED: Only set actual boundary points to infinity
      # # Do NOT set interior points with same values to infinity
      crowdDist[sortedFront[0]] = np.inf   # Minimum boundary
      crowdDist[sortedFront[-1]] = np.inf  # Maximum boundary

      # Skip normalization if all values are identical
      fMax = fitness[sortedFront, obj].max()
      fMin = fitness[sortedFront, obj].min()
      if fMax == fMin:
        continue

      # Calculate normalized distances for interior points
      for i in range(1, numPoints - 1):
        # Skip if already set to infinity (can happen if point is boundary in another objective)
        if crowdDist[sortedFront[i]] != np.inf:
          nextObjValue = fitness[sortedFront[i + 1], obj]
          prevObjValue = fitness[sortedFront[i - 1], obj]
          # Add normalized distance for this objective
          crowdDist[sortedFront[i]] += (nextObjValue - prevObjValue) / (fMax - fMin)

  return crowdDist
