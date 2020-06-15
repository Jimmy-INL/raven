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
  Implementation of AgeBased Selector for Survivor selection of Genetic Algorithm
"""
import numpy as np
from utils import InputData, InputTypes, randomUtils, mathUtils
from . import SurvivorSelectors

class AgeBased(SurvivorSelectors):
  """
    Uses AgeBased Selection approach to select next generation
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
    specs = super(AgeBased, cls).getInputSpecification()
    specs.description = r"""if node is present, indicates that."""
    return specs

  ###############
  # Run Methods #
  ###############
  def Select(self, population, fitnesses):
    pass
  ###################
  # Utility Methods #
  ###################
