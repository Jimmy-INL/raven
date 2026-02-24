import math

import numpy as np
import pytest

from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm
from ravenframework.Optimizers.fitness import fitness


def test_inv_linear_penalizes_constraint_violations(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  constraints = fitness_inputs["constraints"]
  result = fitness.invLinear(
    rlz,
    objVar=["obj"],
    a=[1.0],
    b=[5.0],
    constraintFunction=constraints,
    type=["min"],
  )
  expected = np.array([-1.0, -3.5, -3.0])
  assert np.allclose(result["obj"].data, expected)


def test_feasible_first_uses_worst_objective_for_infeasible(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  constraints = fitness_inputs["constraints"]
  result = fitness.feasibleFirst(
    rlz,
    objVar=["obj"],
    constraintFunction=constraints,
    constraintNum=2,
    a=[1.0],
    b=[2.0],
    type=["min"],
  )
  expected = np.array([-1.0, -2.6, -3.0])
  assert np.allclose(result["obj"].data, expected)


def test_logistic_applies_penalty_and_respects_minimization(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  constraints = fitness_inputs["constraints"]
  result = fitness.logistic(
    rlz,
    objVar=["obj"],
    scale=[-1.0],
    shift=[1.0],
    penalty=[0.5],
    constraintFunction=constraints,
    type=["min"],
  )
  expected = np.array(
    [
      0.5,
      1.0 / (1.0 + math.exp(-1.0)) - 0.5 * 0.3,
      1.0 / (1.0 + math.exp(0.5)) - 0.5 * 0.5,
    ]
  )
  assert np.allclose(result["obj"].data, expected)


def test_logistic_flips_for_maximization(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  result = fitness.logistic(
    rlz,
    objVar=["obj"],
    scale=[1.0],
    shift=[1.0],
    penalty=[0.0],
    type=["max"],
  )
  expected = np.array(
    [
      1.0 / (1.0 + math.exp(-0.0)),
      1.0 / (1.0 + math.exp(-1.0)),
      1.0 / (1.0 + math.exp(0.5)),
    ]
  )
  assert np.allclose(result["obj"].data, expected)


def test_positive_fitness_shift_outputs_non_negative():
  ga = GeneticAlgorithm()
  ga._positiveFitness = True
  ga._positiveFitnessEps = 0.1
  fitness_ds = {
    "obj": np.array([-2.0, -0.5, 0.0]),
  }
  shifted = ga._shiftFitnessForOutput(fitness_ds)
  assert shifted is not None
  assert np.min(shifted["obj"]) >= 0.1


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
