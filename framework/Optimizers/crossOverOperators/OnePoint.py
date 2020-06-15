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

class OnePoint(Crossovers):
  """
    Uses One Point approach to perform Crossover
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
    specs = super(OnePoint, cls).getInputSpecification()
    specs.description = r"""if node is present, indicates that."""
    return specs

  ###############
  # Run Methods #
  ###############
  def onePoint(self,parent1,parent2,crossoverProb,point):
    """
      One Point crossover.
      @ In, parent1, 1D array, parent1 in the current mating process. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
      @ In, parent2, 1D array, parent2 in the current mating process. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
      @ In, crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
      @ In, point, integer, point at which the cross over happens, default is random
      @ Out, child1, 1D array, child1 resulting from the crossover. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
      @ Out, child2, 1D array, child2 resulting from the crossover. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
    """
    nGenes = np.shape(parent1)[1]
    # defaults
    if point is None:
      point = randomUtils.randomIntegers(1,nGenes-1)
    if crossoverProb is None:
      crossoverProb = randomUtils.random(dim=1, samples=1)
    # create children
    if randomUtils.random(dim=1,samples=1) < crossoverProb:
      ## TODO create n children, where n is equal to number of parents
      ## add code here

      for i in range(nGenes):
        if i<point:
          child1[1,i]=parent1[1,i]
          child2[1,i]=parent2[1,i]
        else:
          child1[1,i]=parent2[1,i]
          child2[1,i]=parent1[1,i]
    else:
      # Each child is just a copy of the parents
      child1 = deepcopy(parent1)
      child2 = deepcopy(parent2)
    return child1,child2


  ###################
  # Utility Methods #
  ###################
