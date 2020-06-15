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
  Genetic Algorithm class for global optimization.
  This class contains the API and interface for performing
  Genetic Algorithm-based optimizzation. Multiple strategies for
  mutations, cross-overs, etc. are available.
  Created June,3,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
  References
    ----------
    .. [1]
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
from collections import deque, defaultdict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils, randomUtils, InputData, InputTypes
from .RavenSampled import RavenSampled
# Parent Selectors
from .parentSelectors import knownTypes as parentSelectorsKnownTypes
from .parentSelectors import returnInstance as parentSelectorsReturnInstance
from .parentSelectors import returnClass as parentSelectorsReturnClass
# Cross Overs
from .crossOverOperators import knownTypes as crossOversKnownTypes
from .crossOverOperators import returnInstance as crossOversReturnInstance
from .crossOverOperators import returnClass as crossOversReturnClass
# Mutators
from .mutators import knownTypes as mutatorsKnownTypes
from .mutators import returnInstance as mutatorsReturnInstance
from .mutators import returnClass as mutatorsReturnClass
# Survivor Selectors
from .survivorSelector import knownTypes as survivorSelectorsKnownTypes
from .survivorSelector import returnInstance as survivorSelectorsReturnInstance
from .survivorSelector import returnClass as survivorSelectorsReturnClass
#Internal Modules End--------------------------------------------------------------------------------

class GeneticAlgorithm(RavenSampled):
  """
    This class performs Genetic Algorithm optimization ...
  """
  convergenceOptions = {'objective': r""" provides the desired value for the convergence criterion of the objective function
                        ($\epsilon^{obj}$), i.e., convergence is reached when: $$ |newObjevtive - oldObjective| \le \epsilon^{obj}$$.
                        \default{1e-6}, if no criteria specified"""}
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    RavenSampled.__init__(self)
    self._parentSelection = None                                  # mechanism for parent selection
    self._convergenceCriteria = defaultdict(mathUtils.giveZero)  # names and values for convergence checks
    self._acceptHistory = {}                                    # acceptability
    self._acceptRerun = {}                                      # by traj, if True then override accept for point rerun
    self._convergenceInfo = {}                                   # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 0                                # consecutive persistence required to mark convergence
    self.needDenormalized()

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
    specs = super(GeneticAlgorithm, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{GeneticAlgorithm} optimizer is a metaheuristic approach
                            to perform a global search in large design spaces. The methodology rose
                            from the process of natural selection, and like others  in the large class of the evolutionary algorithms, it utilizes genetic operations such as selection crossover and mutations to avoid being stuck in local minima and hence facilitates
                            finding the global minima. More information can be found in:
                            Holland, John H. "Genetic algorithms." Scientific american 267.1 (1992): 66-73."""

    # GA Params
    GAparams = InputData.parameterInputFactory('GAparams', strictMode=True,
        printPriority=108,
        descr=r""" Genetic Algorithm Parameters.""")
    specs.addSub(GAparams)
    # Population Size
    populationSize = InputData.parameterInputFactory('populationSize', strictMode=True,
        contentType=InputTypes.IntegerType,
        printPriority=108,
        descr=r"""The number of chromosomes in each population.""")
    GAparams.addSub(populationSize)
    # Parent Selection
    parentSelection = InputData.parameterInputFactory('parentSelection', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a node containing the criterion based on which the parents are selected. This can be a. a fitness proportionate selection such as Roulette Wheer, Stochastic Universal Sampling,
                  b. Tournament, c. Rank, or d. Random selection""")
    GAparams.addSub(parentSelection)
    ## get specs for each ParentSelection subclass, and add them to this class's options
    for selector in parentSelectorsKnownTypes():
      subSpecs = parentSelectorsReturnClass(selector, cls).getInputSpecification()
      GAparams.addSub(subSpecs)

    # Reproduction
    reproduction = InputData.parameterInputFactory('reproduction', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the reproduction methods.
                  This accepts subnodes that specifies the types of crossover and mutation.""")
    reproduction.addParam("nParents", InputTypes.IntegerType, True)
    # specs.addSub(reproduction)
    # 1.  Crossover
    crossover = InputData.parameterInputFactory('crossover', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented crossover mechanisms.
                  This includes: a.    One Point Crossover,
                                 b.    MultiPoint Crossover,
                                 c.    Uniform Crossover,
                                 d.    Whole Arithmetic Recombination, or
                                 e.    Davis’ Order Crossover.""")
    crossover.addParam("crossoverPoint", InputTypes.IntegerType, True)
    ## get specs for each crossover subclass, and add them to this class's options
    for crossover in crossOversKnownTypes():
      subSpecs = crossOversReturnClass(crossover, cls).getInputSpecification()
      crossover.addSub(subSpecs)
    reproduction.addSub(crossover)
    # specs.addSub(crossover)
    # 2.  Mutation
    mutation = InputData.parameterInputFactory('mutation', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented mutation mechanisms.
                  This includes: a.    Bit Flip,
                                 b.    Random Resetting,
                                 c.    Swap,
                                 d.    Scramble, or
                                 e.    Inversion.""")
    ## get specs for each mutator subclass, and add them to this class's options
    for mutator in mutatorsKnownTypes():
      subSpecs = mutatorsReturnClass(mutator, cls).getInputSpecification()
      mutation.addSub(subSpecs)
    reproduction.addSub(mutation)
    # specs.addSub(mutation)
    GAparams.addSub(reproduction)

    # Survivor Selection
    survivorSelection = InputData.parameterInputFactory('survivorSelection', strictMode=True,
        contentType=InputTypes.StringType,
        printPriority=108,
        descr=r"""a subnode containing the implemented servivor selection mechanisms.
                  This includes: a.    AgeBased, or
                                 b.    Fitness Based.""")
    ## get specs for each survivorSelector subclass, and add them to this class's options
    for survivor in survivorSelectorsKnownTypes():
      subSpecs = survivorSelectorsReturnClass(survivor, cls).getInputSpecification()
      survivorSelection.addSub(subSpecs)
    GAparams.addSub(survivorSelection)

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the desired convergence criteria for the optimization algorithm.
              Note that convergence is met when any one of the convergence criteria is met. If no convergence
              criteria are given, then the defaults are used.""")
    specs.addSub(conv)
    for name,descr in cls.convergenceOptions.items():
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType,descr=descr,printPriority=108  ))

    # Presistance
    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType,
        printPriority = 109,
        descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
              is considered fully converged. This helps in preventing early false convergence."""))
    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, cls, the class for which we are retrieving the solution export
      @ Out, ok, dict, {varName: description} for valid solution export variable names
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(GeneticAlgorithm, cls).getSolutionExportVariableNames()
    new = {}
    # new = {'': 'the size of step taken in the normalized input space to arrive at each optimal point'}
    new['conv_{CONV}'] = 'status of each given convergence criteria'
    ok.update(new)
    return ok

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)

    # GAparams
    GAparamsNode = paramInput.findFirst('GAparams')
    for sub in GAparamsNode.subparts:
      if sub.name=='reproduction':
        for subsub in sub.subparts:
          setattr(self,str('_'+subsub.name),subsub.value)
          # setattr(self,str('_'+subsub.name+'Instance'),subsub.name+'ReturnInstance(self,'+)
      else:
        setattr(self,str('_'+sub.name),sub.value)
      # self._parentSelectionInstance =

    # Convergence Criterion
    convNode = paramInput.findFirst('convergence')
    if convNode is not None:
      for sub in convNode.subparts:
        if sub.getName() == 'persistence':
          self._requiredPersistence = sub.value
        else:
          self._convergenceCriteria[sub.name] = sub.value
    if not self._convergenceCriteria:
      self.raiseAWarning('No convergence criteria given; using defaults.')
      self._convergenceCriteria['objective'] = 1e-6
    # same point is ALWAYS a criterion
    self._convergenceCriteria['samePoint'] = -1 # For simulated Annealing samePoint convergence
                                                # should not be one of the stopping criteria
    # set persistence to 1 if not set
    if self._requiredPersistence is None:
      self.raiseADebug('No persistence given; setting to 1.')
      self._requiredPersistence = 1

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    for traj, init in enumerate(self._initialValues):
      self._submitRun(init,traj,self.getIteration(traj))

  ###############
  # Run Methods #
  ###############
  # abstract methods:
  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      This is called by localFinalizeActualSampling, and hence should contain the main skeleton.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization, In case of bath handling rlz is a dictionary of
            dictionaries representing the population with the corresponding evaluations.
      @ In, optVal, float, value of objective variable (corrected for min/max)
      @ Out, None
    """
    ## TODO the whole skeleton should be here, this should be calling all classes and _private methods.
    traj = info['traj']
    info['optVal'] = rlz[self._objectiveVar]
    self.incrementIteration(traj)
    info['step'] = self.counter

    # separate population from model evaluations
    population = np.zeros((self._populationSize,len(self.toBeSampled)))
    fitness = np.zeros(len(self.toBeSampled))
    fitness = rlz[self._objectiveVar]*np.random.random(self._populationSize) # All np.random should be replaced with randomUtils.random etc.
    chromosome = rlz.copy()
    chromosome.pop(self._objectiveVar)
    chromosome = list(chromosome.values())
    for i in range(self._populationSize):
      population[i] = np.random.choice(chromosome,size=len(self.toBeSampled),replace=False)

    # Select Parents
    parentSelectorsKnownTypes()
    # reproduce

    # crossover

    # Mutation


  def _submitRun(self, point, traj, step, moreInfo=None):
    """
      Submits a single run with associated info to the submission queue
      @ In, point, dict, point to submit
      @ In, traj, int, trajectory identifier
      @ In, step, int, iteration number identifier
      @ In, moreInfo, dict, optional, additional run-identifying information to track
      @ Out, None
    """
    info = {}
    if moreInfo is not None:
      info.update(moreInfo)
    info.update({'traj': traj,
                  'step': step
                })
    # NOTE: explicit constraints have been checked before this!
    self.raiseADebug('Adding run to queue: {} | {}'.format(self.denormalizeData(point), info))
    self._submissionQueue.append((point, info))
  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  def checkConvergence(self, traj):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ Out, None
    """
    pass

  def _checkAcceptability(self, traj, opt, optVal):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
    """
    pass

  def _updateConvergence(self, traj, rlz, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, acceptable, str, condition of point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    pass

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
  pass

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    pass

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
  pass

  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    pass
  # END constraint handling
  # * * * * * * * * * * * *

  # def _updateSolutionExport(self, traj, rlz, acceptable):
  #   """
  #     Stores information to the solution export.
  #     @ In, traj, int, trajectory which should be written
  #     @ In, rlz, dict, collected point
  #     @ In, acceptable, bool, acceptability of opt point
  #     @ Out, None
  #   """
  #   pass
  def needDenormalized(self):
    """
      Determines if the currently used algorithms should be normalizing the input space or not
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    # overload as needed in inheritors
    return True