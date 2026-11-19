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
  Traveling salesman problem (TSP) with 10 cities, single- and multi-objective.

  The decision variables x1, ..., x10 are the city visited at each stop of a closed tour, so
  {x1, ..., x10} must be a permutation of 1..10 (in RAVEN: all ten variables share one
  UniformDiscrete distribution with the withoutReplacement strategy). The tour returns from the
  last city to the first.

  Each objective is the Euclidean length of the same tour on a different map of the 10 cities:
  lengthA on map A, lengthB on map B, lengthC on map C. The three maps are independent, like
  distance, travel time and cost between the same cities, so the objectives conflict. This is the
  standard multi-objective TSP construction (e.g. the kroAB100 and kroABC100 instances of Lust and
  Teghem, 2010), at a size small enough to enumerate.

  Exact solutions, found by enumerating all 9!/2 = 181440 distinct tours:
    - min lengthA = 254.3848, tour 1-8-2-4-9-7-6-3-5-10 (unique up to start city and direction).
    - the (lengthA, lengthB) Pareto front has 17 points, from (254.3848, 369.6443) to
      (508.5362, 219.2863); see PARETO_FRONT_AB.

  The coordinates are integers in [0, 100], drawn with numpy.random.RandomState(2).
"""
import numpy as np

CITIES_A = np.array([[40, 15], [72, 22], [43, 82], [75, 7], [34, 49],
                     [95, 75], [85, 47], [63, 31], [90, 20], [37, 39]], dtype=float)
CITIES_B = np.array([[67, 4], [42, 51], [38, 33], [58, 67], [69, 88],
                     [68, 46], [70, 95], [83, 31], [66, 80], [52, 76]], dtype=float)
CITIES_C = np.array([[50, 4], [90, 63], [79, 49], [39, 46], [8, 50],
                     [15, 8], [17, 22], [73, 57], [90, 62], [83, 96]], dtype=float)

NUM_CITIES = 10
OPTIMAL_LENGTH_A = 254.3848

# exact (lengthA, lengthB) Pareto front
PARETO_FRONT_AB = np.array([
  [254.3848, 369.6443], [273.9347, 362.2091], [277.4298, 358.3601], [293.5702, 344.4968],
  [310.2346, 330.9831], [320.3174, 328.5138], [323.3743, 313.9778], [323.5627, 302.0607],
  [334.9418, 295.3805], [335.1511, 265.0461], [349.8054, 258.5741], [374.7679, 246.6038],
  [398.7926, 241.4217], [408.4809, 227.3448], [425.2731, 222.9559], [434.7100, 220.9027],
  [508.5362, 219.2863]])

def tourLength(tour, cities):
  """
    Length of the closed tour through the given cities.
    @ In, tour, list(int), city numbers 1..NUM_CITIES in visiting order
    @ In, cities, np.array, (NUM_CITIES, 2) city coordinates on one map
    @ Out, length, float, total Euclidean length, including the leg back to the first city
  """
  points = cities[np.asarray(tour, dtype=int) - 1]
  legs = points - np.roll(points, -1, axis=0)
  return float(np.sqrt((legs**2).sum(axis=1)).sum())

def evaluate(inputs):
  """
    Evaluate the tour given by x1..x10 on the three maps.
    @ In, inputs, dict, the variables x1..x10 (city at each stop)
    @ Out, lengths, tuple(float), (lengthA, lengthB, lengthC)
  """
  tour = [int(round(float(np.atleast_1d(inputs['x{}'.format(i)])[0]))) for i in range(1, NUM_CITIES + 1)]
  if sorted(tour) != list(range(1, NUM_CITIES + 1)):
    raise ValueError('travelingSalesman: x1..x{} must be a permutation of 1..{}, got {}'.format(NUM_CITIES, NUM_CITIES, tour))
  return tourLength(tour, CITIES_A), tourLength(tour, CITIES_B), tourLength(tour, CITIES_C)

def run(self, inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, inputs, dict, additional inputs
    @ Out, None
  """
  self.lengthA, self.lengthB, self.lengthC = evaluate(inputs)
