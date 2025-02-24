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
  Implementation of survivorSelctors (Elitism) for new generation
  selection process of Genetic Algorithm. Currently the implemented
  survivorSelctors algorithms are:
  1.  ageBased
  2.  fitnessBased

  Created June,16,2020
  @authors: Mohammad Abdo, Junyung Kim, Diego Mandelli, Andrea Alfonsi
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from ravenframework.utils import frontUtils
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils.gaUtils import dataArrayToDict, datasetToDataArray
# Internal Modules End------------------------------------------------------------------------------

# @profile

def ageBased(newRlz,**kwargs):
  """
    ageBased survivorSelection mechanism for new generation selection.
    It replaces the oldest parents with the new children regardless of the fitness.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          age, list, age list for each chromosome of the previous population
          variables, list of variable names to be sampled
          fitness, xr.DataArrays, fitness of the previous generation
          offSpringsFitness, xr.DataArray, fitness of each new child, i.e., np.shape(offSpringsFitness) = nChildren x nGenes
          population, xr.DataArray, population from previous generation
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitness, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
  """
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs.keys() or kwargs['age'] is None):
    popAge = [0] * popSize
  else:
    popAge = kwargs['age']
  # offSpringsFitness = np.atleast_1d(kwargs['offSpringsFitness'])
  offSpringsFitness = datasetToDataArray(kwargs['offSpringsFitness'], list(kwargs['offSpringsFitness'].keys())).data
  offSprings = xr.DataArray(np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose()),
                            dims=['chromosome','Gene'],
                            coords={'chromosome':np.arange(np.shape(np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose()))[0]),
                                    'Gene': kwargs['variables']})
  population = np.atleast_2d(kwargs['population'].data)
  # popFitness = np.atleast_1d(kwargs['fitness'].data)
  popFitness = datasetToDataArray(kwargs['fitness'], list(kwargs['fitness'].keys())).data
  # sort population, popFitness according to age
  sortedAge,sortedPopulation,sortedFitness = zip(*[[x,y,z] for x,y,z in sorted(zip(popAge,population,popFitness),key=lambda x: (x[0], -x[2]))])# if equal age then use descending fitness
  sortedAge,sortedPopulation,sortedFitness = list(sortedAge),np.atleast_1d(list(sortedPopulation)),np.atleast_1d(list(sortedFitness))
  newPopulation = sortedPopulation
  newFitness    = np.squeeze(sortedFitness)
  newAge = list(map(lambda x:x+1, sortedAge))
  newPopulation[-1:-np.shape(offSprings)[0]-1:-1] = offSprings
  newFitness[-1:-np.shape(offSprings)[0]-1:-1] = np.squeeze(offSpringsFitness)
  newAge[-1:-np.shape(offSprings)[0]-1:-1] = [0]*np.shape(offSprings)[0]
  # converting back to DataArrays
  newPopulation = xr.DataArray(newPopulation,
                               dims=['chromosome','Gene'],
                               coords={'chromosome':np.arange(np.shape(newPopulation)[0]),
                                       'Gene': kwargs['variables']})
  newFitnessDS = xr.Dataset()
  newFitnessDS[kwargs['objVar']] = xr.DataArray(newFitness,
                               dims=['chromosome'],
                               coords={'chromosome':np.arange(np.shape(newFitness)[0])})
  return newPopulation,newFitnessDS,newAge,kwargs['popObjectiveVal']


# @profile
def fitnessBased(newRlz,**kwargs):
  """
    fitnessBased survivorSelection mechanism for new generation selection
    It combines the parents and children/offsprings then keeps the fittest individuals
    to revert to the same population size.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this survivor slection method:
          age, list, ages of each chromosome in the population of the previous generation
          offSpringsFitness, xr.DataArray, fitness of each new child, i.e., np.shape(offSpringsFitness) = nChildren x nGenes
          variables
          population
          fitness
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitness, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
  """
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs.keys() or kwargs['age'] is None):
    popAge = [0] * popSize
  else:
    popAge = kwargs['age']

  offSpringsFitness = datasetToDataArray(kwargs['offSpringsFitness'], list(kwargs['offSpringsFitness'].keys())).data
  offSpringsFitness = np.array([item for sublist in offSpringsFitness for item in sublist])
  offSprings = np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose().data)
  population = np.atleast_2d(kwargs['population'].data)
  popFitness = datasetToDataArray(kwargs['fitness'], list(kwargs['fitness'].keys())).data
  popFitness = popFitness.reshape((popFitness.size,))
  newPopulation = population
  newFitness = popFitness
  newAge = list(map(lambda x:x+1, popAge))
  newPopulationMerged = np.concatenate([newPopulation,offSprings])
  newFitness = np.concatenate([newFitness,offSpringsFitness])
  newAge.extend([0]*len(offSpringsFitness))

  # sort population, popFitness according to age
  sortedFitness,sortedAge,sortedPopulation = zip(*[(x,y,z) for x,y,z in sorted(zip(newFitness,newAge,newPopulationMerged),reverse=True,key=lambda x: (x[0], -x[1]))])
  sortedFitnessT,sortedAgeT,sortedPopulationT = np.atleast_1d(list(sortedFitness)),list(sortedAge),np.atleast_1d(list(sortedPopulation))
  newPopulationSorted = sortedPopulationT[:-len(offSprings)]
  newFitness = sortedFitnessT[:-len(offSprings)]
  newAge = sortedAgeT[:-len(offSprings)]

  newPopulationArray = xr.DataArray(newPopulationSorted,
                                    dims=['chromosome','Gene'],
                                    coords={'chromosome':np.arange(np.shape(newPopulationSorted)[0]),
                                            'Gene': kwargs['variables']})
  newFitnessDS = xr.Dataset()
  newFitnessDS[kwargs['objVar']] = xr.DataArray(newFitness,
                            dims=['chromosome'],
                            coords={'chromosome':np.arange(np.shape(newFitness)[0])})
  return newPopulationArray,newFitnessDS,newAge,kwargs['popObjectiveVal']

# @profile
def rankNcrowdingBased(offsprings, **kwargs):
  """
  rankNcrowdingBased survivorSelection mechanism for new generation selection
  It combines the parents and children/offsprings then calculates their rank and crowding distance.
  After having ranks and crowding distance, it keeps the lowest ranks (and highest crowding distance if individuals have same rank).

  @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
  @ In, useElitism, bool, whether to use elitism with diversity preservation.
  @ In, similarityThreshold, float, threshold for determining similarity in diversity preservation.
  @ In, kwargs, dict, dictionary of parameters for this survivor selection method:
        variables
        age
        population
        popObjectiveVal
        offObjectiveVal
        popFit
        offFit
        popConstV
        offConstV

        useElitism
          The default is True, cases when the user might want to make oit false:
          - Highly Dynamic Environments:
             Context: In environments where the optimization landscape changes frequently and significantly, maintaining a diverse population can be more important than preserving the best solutions from the previous generation.
             Rationale: Elitism might cause the algorithm to retain solutions that are no longer optimal in the new context, leading to slower adaptation to changes.
          - Initial Exploration Phase:
              Context: During the initial stages of the evolutionary process, exploration of the solution space is crucial to avoid premature convergence to suboptimal regions.
              Rationale: Not using elitism can encourage a broader search of the solution space, allowing more diverse solutions to be explored before focusing on refinement.
          - High Redundancy in Solutions:
              Context: In problems where many solutions are similar or redundant, elitism might lead to a population dominated by similar solutions.
              Rationale: Disabling elitism can help maintain a more diverse set of solutions, which might be beneficial for avoiding local optima and finding better overall solutions.
          - Noisy Evaluation Functions:
              Context: In scenarios where the fitness evaluation has a high degree of noise or uncertainty, preserving the top solutions might not be reliable.
              Rationale: Without elitism, the algorithm may be more robust to noise by continuously sampling new solutions and averaging out the noise over many generations.
          - When Computational Resources are Constrained:
              Context: Situations where computational resources (like time or memory) are limited might not afford the luxury of maintaining an elite set of solutions.
              Rationale: Disabling elitism can reduce computational overhead and allow the algorithm to run more efficiently under constrained resources.

        similarityThreshold:
          Increasing similarity_threshold
            Effect: When you increase the similarity_threshold, it means that solutions need to be more different from each other to be considered distinct. This can lead to higher diversity in the population, as it prevents the algorithm from keeping solutions that are too similar.
            When to Increase:
              If the population is converging prematurely and you want to explore a wider range of solutions.
              If there is a risk of getting stuck in local optima and you need to encourage more exploration.
          Decreasing similarity_threshold:
            Effect: When you decrease the similarity_threshold, solutions can be more similar to one another while still being considered distinct. This can lead to less diversity in the population, focusing more on exploitation of the current best solutions.
            When to Decrease:
              If the algorithm is not converging and you want to focus more on refining current solutions.
              If you have a very diverse population and need to start converging towards optimal solutions.
  @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
  @ Out, newRank, xr.DataArray, rank of each chromosome in the new population
  @ Out, newAge, list, integer age of each chromosome
  @ Out, newCD, xr.DataArray, crowding distance of each chromosome in the new population.
  @ Out, newObjectivesP, list of lists, float value of the objectives
  @ Out, newFitnessSet, xr.DataSet, objectives of the chromosome
  @ Out, newConstV, xr.DataArray, includes the ConstEvaluation
  """
  if 'useElitism' not in kwargs.keys():
    useElitism = True
  if 'similarityThreshold' not in kwargs.keys():
    similarityThreshold = 0.9
  self = kwargs['self']
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs or kwargs['age'] is None):
    popAge = [0] * popSize
  else:
    popAge = kwargs['age']
  population = np.atleast_2d(kwargs['population'].data)
  offSprings = np.atleast_2d(offsprings[kwargs['variables']].to_array().transpose().data)
  popObjectiveVal = kwargs['popObjectiveVal']
  offObjectiveVal = kwargs['offObjectiveVal']
  popFit = kwargs['popFit']
  popFitArray = []
  offFit = kwargs['offFit']
  offFitArray = []
  for i in list(popFit.keys()):  # NOTE popFit.keys() and offFit.keys() must be same.
    popFitArray.append(popFit[i].data.tolist())
    offFitArray.append(offFit[i].data.tolist())
  # Combine parent and offspring data and population
  newFitMerged = np.array([i + j for i, j in zip(popFitArray, offFitArray)])
  newFitMergedPair = [list(ele) for ele in list(zip(*newFitMerged))]
  popConstV = kwargs['popConstV'].data
  offConstV = kwargs['offConstV'].data
  newConstVMerged = np.vstack([popConstV, offConstV])
  newObjectivesMerged = np.array([i + j for i, j in zip(popObjectiveVal, offObjectiveVal)])
  newObjectivesMergedPair = [list(ele) for ele in list(zip(*newObjectivesMerged))]
  # Calculate nondominated fronts
  newPopRank = frontUtils.rankNonDominatedFrontiers(np.array(newFitMergedPair), isFitness=True)
  newPopRank = xr.DataArray(newPopRank,
                            dims=['rank'],
                            coords={'rank': np.arange(np.shape(newPopRank)[0])})
  # Calculate crowding distance
  newPopCD = frontUtils.crowdingDistance(rank=newPopRank, popSize=len(newPopRank), fitness=np.array(newFitMergedPair))
  newPopCD = xr.DataArray(newPopCD,
                          dims=['CrowdingDistance'],
                          coords={'CrowdingDistance': np.arange(np.shape(newPopCD)[0])})
  newAge = list(map(lambda x: x + 1, popAge))
  newPopulationMerged = np.concatenate([population, offSprings])
  newAge.extend([0] * len(offSprings))
  # Sort in rank/crowd comparison order
  sortedRank, sortedCD, sortedAge, sortedPopulation, sortedFit, sortedObjectives, sortedConstV = \
      zip(*[(x, y, z, i, j, k, a) for x, y, z, i, j, k, a in
            sorted(zip(newPopRank.data, newPopCD.data, newAge, newPopulationMerged.tolist(), newFitMergedPair, newObjectivesMergedPair, newConstVMerged), reverse=False, key=lambda x: (x[0], -x[1], x[4], x[3]))])
  _, _, sortedAgeT, sortedPopulationT, sortedFitT, sortedObjectivesT, sortedConstVT = \
      np.atleast_1d(list(sortedRank)), list(sortedCD), list(sortedAge), np.atleast_1d(list(sortedPopulation)), np.atleast_1d(list(sortedFit)), np.atleast_1d(list(sortedObjectives)), np.atleast_1d(list(sortedConstV))
  if useElitism:
    # Diversity control to ensure uniqueness
    unique_population = []
    unique_fitnesses = []
    unique_objectives = []
    unique_constV = []
    unique_age = []
    for i in range(len(sortedPopulationT)):
      is_unique = True
      for j in range(len(unique_population)):
        if np.mean(sortedPopulationT[i] == unique_population[j]) > similarityThreshold:
          is_unique = False
          break
      if is_unique:
        unique_population.append(sortedPopulationT[i])
        unique_fitnesses.append(sortedFitT[i])
        unique_objectives.append(sortedObjectivesT[i])
        unique_constV.append(sortedConstVT[i])
        unique_age.append(sortedAgeT[i])
    if len(unique_population) < popSize:
      repeated_indices = [i for i in range(len(sortedPopulationT)) if i not in unique_population]
      newChildren = self._mutationInstance(offSprings=np.array(sortedPopulationT)[repeated_indices, :],
                                           distDict=self.distDict,
                                           locs=self._mutationLocs,
                                           mutationProb=self._mutationProb,
                                           variables=list(self.toBeSampled))
      for idx, new_child in zip(repeated_indices, newChildren.data):
        unique_population.append(new_child)
        unique_fitnesses.append(self._evaluate_fitness(new_child))
        unique_objectives.append(sortedObjectivesT[idx])
        unique_constV.append(sortedConstVT[idx])
        unique_age.append(0)
    # Select the best elements
    newPopulation = unique_population[:popSize]
    newObjectives = unique_objectives[:popSize]
    newFit = np.array(unique_fitnesses[:popSize])
    newAge = unique_age[:popSize]
    newConstV = unique_constV[:popSize]
  else:
    # Choose the best elements without elitism
    newPopulation = sortedPopulationT[:popSize]
    newObjectives = sortedObjectivesT[:popSize]
    newFit = sortedFitT[:popSize]
    newAge = sortedAgeT[:popSize]
    newConstV = sortedConstVT[:popSize]
  newRank = frontUtils.rankNonDominatedFrontiers(newFit, isFitness=True)
  newRank = xr.DataArray(newRank,
                         dims=['rank'],
                         coords={'rank': np.arange(np.shape(newRank)[0])})
  newObjectivesP = [list(ele) for ele in list(zip(*newObjectives))]
  newCD = frontUtils.crowdingDistance(rank=newRank, popSize=len(newRank), fitness=newFit)
  newCD = xr.DataArray(newCD,
                       dims=['CrowdingDistance'],
                       coords={'CrowdingDistance': np.arange(np.shape(newCD)[0])})
  for i in range(len(list(popFit.keys()))):
    fitness = xr.DataArray(newFit[:, i],
                           dims=['chromosome'],
                           coords={'chromosome': np.arange(len(newFit[:, i]))})
    if i == 0:
      newFitnessSet = fitness.to_dataset(name=list(popFit.keys())[i])
    else:
      newFitnessSet[list(popFit.keys())[i]] = fitness
  newPopulationArray = xr.DataArray(newPopulation,
                                    dims=['chromosome', 'Gene'],
                                    coords={'chromosome': np.arange(np.shape(newPopulation)[0]),
                                            'Gene': kwargs['variables']})
  newConstV = xr.DataArray(newConstV,
                           dims=['chromosome', 'ConstEvaluation'],
                           coords={'chromosome': np.arange(np.shape(newPopulation)[0]),
                                   'ConstEvaluation': np.arange(np.shape(newConstV)[1])})
  return newPopulationArray, newRank, newAge, newCD, newObjectivesP, newFitnessSet, newConstV

__survivorSelectors = {}
__survivorSelectors['ageBased'] = ageBased
__survivorSelectors['fitnessBased'] = fitnessBased
__survivorSelectors['rankNcrowdingBased'] = rankNcrowdingBased

def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __survivorSelectors:
    cls.raiseAnError (IOError, "{} is not an valid option for survivor selector. Please review the spelling of the survivor selector. ".format(name))
  return __survivorSelectors[name]
