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
  Unit tests for the GA crossover operators (shape, gene provenance, and reproducibility).
"""
import numpy as np
import pytest
import xarray as xr

from ravenframework.Optimizers.crossOverOperators import crossovers
from ravenframework.utils import randomUtils


@pytest.fixture
def simple_parents():
  data = np.array(
    [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2],
    ]
  )
  parents = xr.DataArray(
    data,
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(3), 'Gene': ['x1', 'x2', 'x3', 'x4']},
  )
  return parents


def test_one_point_crossover_uses_provided_split(simple_parents, monkeypatch):
  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return 0.0

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  children = crossovers.onePointCrossover(
    simple_parents,
    crossoverProb=0.5,
    points=[2],
    variables=list(simple_parents.coords['Gene'].values),
  )
  expected = np.array(
    [
      [0.1, 0.2, 0.7, 0.8],
      [0.5, 0.6, 0.3, 0.4],
      [0.1, 0.2, 1.1, 1.2],
      [0.9, 1.0, 0.3, 0.4],
      [0.5, 0.6, 1.1, 1.2],
      [0.9, 1.0, 0.7, 0.8],
    ]
  )
  np.testing.assert_allclose(children.values, expected)


def test_uniform_crossover_method_swaps_by_probability(monkeypatch):
  parent1 = np.array([0.0, 0.1, 0.2])
  parent2 = np.array([1.0, 1.1, 1.2])
  probs = iter([0.0, 0.6, 0.4])

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return next(probs)

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  child1, child2 = crossovers.uniformCrossoverMethod(parent1, parent2, 0.5)
  assert np.allclose(child1, np.array([1.0, 0.1, 1.2]))
  assert np.allclose(child2, np.array([0.0, 1.1, 0.2]))


def test_uniform_crossover_for_parent_pairs(simple_parents, monkeypatch):
  calls = iter([0.0] * 12)

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return next(calls)

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  children = crossovers.uniformCrossover(
    simple_parents,
    crossoverProb=0.5,
  )
  assert children.shape == (6, 4)
  assert np.all(children.values[:2, 0] == np.array([0.5, 0.1]))


def test_two_points_crossover_uses_sampled_points(simple_parents, monkeypatch):
  def fake_choice(array, size=1, replace=True, engine=None):
    return [1, 3]

  monkeypatch.setattr(randomUtils, 'randomChoice', fake_choice)
  children = crossovers.twoPointsCrossover(
    simple_parents,
    crossoverProb=1.0,
  )
  expected_first_pair = np.array(
    [
      [0.1, 0.6, 0.7, 0.4],
      [0.5, 0.2, 0.3, 0.8],
    ]
  )
  np.testing.assert_allclose(children.values[:2], expected_first_pair)


def test_two_points_crossover_rejects_short_chromosomes(simple_parents):
  with pytest.raises(ValueError):
    crossovers.twoPointsCrossover(
      simple_parents[:, :2],
      crossoverProb=1.0,
    )


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)


@pytest.fixture
def permutation_parents():
  parents = xr.DataArray(
    np.array([[1, 2, 3, 4, 5, 6, 7, 8],
              [8, 6, 4, 2, 7, 5, 3, 1]], dtype=float),
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(2), 'Gene': ['x{}'.format(i) for i in range(1, 9)]},
  )
  return parents


def test_order_crossover_method_matches_davis_example():
  parent1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  parent2 = np.array([8, 6, 4, 2, 7, 5, 3, 1])
  np.testing.assert_array_equal(crossovers.orderCrossoverMethod(parent1, parent2, 2, 5),
                                [2, 7, 3, 4, 5, 1, 8, 6])
  np.testing.assert_array_equal(crossovers.orderCrossoverMethod(parent2, parent1, 2, 5),
                                [3, 5, 4, 2, 7, 6, 8, 1])


def test_order_crossover_method_children_are_permutations_keeping_the_slice():
  rng = np.random.RandomState(0)
  for _ in range(500):
    nGenes = rng.randint(2, 13)
    parent1 = rng.permutation(nGenes)
    parent2 = rng.permutation(nGenes)
    locL, locU = sorted(rng.choice(nGenes + 1, size=2, replace=False))
    child = crossovers.orderCrossoverMethod(parent1, parent2, locL, locU)
    assert sorted(child.tolist()) == list(range(nGenes))
    np.testing.assert_array_equal(child[locL:locU], parent1[locL:locU])


def test_order_crossover_method_full_slice_copies_parent():
  parent1 = np.array([3, 1, 2, 4])
  parent2 = np.array([4, 3, 2, 1])
  np.testing.assert_array_equal(crossovers.orderCrossoverMethod(parent1, parent2, 0, 4), parent1)


def test_order_crossover_uses_provided_points(permutation_parents, monkeypatch):
  monkeypatch.setattr(randomUtils, 'random', lambda dim=1, samples=1, keepMatrix=False, engine=None: 0.0)
  children = crossovers.orderCrossover(permutation_parents, crossoverProb=0.9, points=[2, 5],
                                       variables=list(permutation_parents.coords['Gene'].values))
  np.testing.assert_array_equal(children.values, [[2, 7, 3, 4, 5, 1, 8, 6],
                                                  [3, 5, 4, 2, 7, 6, 8, 1]])
  assert list(children.coords['Gene'].values) == list(permutation_parents.coords['Gene'].values)


def test_order_crossover_copies_parents_when_not_recombined(permutation_parents, monkeypatch):
  monkeypatch.setattr(randomUtils, 'random', lambda dim=1, samples=1, keepMatrix=False, engine=None: 1.0)
  children = crossovers.orderCrossover(permutation_parents, crossoverProb=0.5, points=[2, 5],
                                       variables=list(permutation_parents.coords['Gene'].values))
  np.testing.assert_array_equal(children.values, permutation_parents.values)


def test_order_crossover_sampled_points_give_permutations(monkeypatch):
  randomUtils.randomSeed(42)
  data = np.array([np.random.RandomState(seed).permutation(10) + 1 for seed in range(4)], dtype=float)
  parents = xr.DataArray(data, dims=['chromosome', 'Gene'],
                         coords={'chromosome': np.arange(4), 'Gene': ['x{}'.format(i) for i in range(1, 11)]})
  children = crossovers.orderCrossover(parents, crossoverProb=1.0, points=None,
                                       variables=list(parents.coords['Gene'].values))
  assert children.shape == (12, 10)
  for child in children.values:
    assert sorted(child.astype(int).tolist()) == list(range(1, 11))


@pytest.mark.parametrize('points', [[3], [5, 2], [0, 9], [-1, 4]])
def test_order_crossover_rejects_bad_points(permutation_parents, points):
  with pytest.raises(ValueError):
    crossovers.orderCrossover(permutation_parents, crossoverProb=1.0, points=points,
                              variables=list(permutation_parents.coords['Gene'].values))


def test_order_crossover_is_registered():
  assert crossovers.returnInstance(None, 'orderCrossover') is crossovers.orderCrossover
