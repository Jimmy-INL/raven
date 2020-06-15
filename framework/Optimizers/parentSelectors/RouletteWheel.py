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
  Implementation of Roulette Selector for parent selection of Genetic Algorithm
"""
import numpy as np
from utils import InputData, InputTypes, randomUtils, mathUtils
from .ParentSelectors import ParentSelectors

class RouletteWheel(ParentSelectors):
  """
    Uses Roulette Wheel Selection approach to select parents
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(RouletteWheel, cls).getInputSpecification()
    specs.description = r"""if node is present, indicates that."""
    return specs

  ###############
  # Run Methods #
  ###############
  def Select(self, population, fitnesses):
    """
      Roulette Selection mechanism for parent selection
      @ In, population, 2darray, all chromosomes (idividuals) candidate to be parents, i.e. np.shape(population) = populationSize x nGenes.
      @ In, fitness, 1darray, fitness of each chromosome, i.e., len(fitness) = populationSize
      @ Out, selectedParents,2darray, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
    """
    selectionProb = fitnesses/np.sum(fitnesses)
    # imagine a wheel that is partitioned according to the selection
    # probabilities

    # set a random pointer
    roulettePointer = randomUtils.random(dim=1, samples=1)
    # Rotate the wheel
    counter = 0
    # intialize Probability
    sumProb = selectionProb[counter]
    while sumProb < roulettePointer:
      counter += 1
      sumProb += selectionProb[counter]
    selectedParent = population[counter]
    return selectedParent


  ###################
  # Utility Methods #
  ###################
