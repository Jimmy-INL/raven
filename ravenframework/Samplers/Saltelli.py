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
  This module contains the Saltelli cross-sampling strategy used to estimate
  variance-based (Sobol') first- and total-order sensitivity indices.

  Unlike the ``Sobol`` sampler (which builds an HDMR / sparse-grid ROM), this
  sampler generates the classical Saltelli A / B / AB_i evaluation matrix
  directly, so that the raw model runs can be fed to the ``SobolIndices``
  post-processor for the Jansen/Saltelli estimators. The resulting run count is
  ``N*(D+2)`` for first/total-order indices, or ``N*(2D+2)`` when second-order
  indices are also requested (D = number of sampled variables, N = base sample).

  Created on 2026-08-14
  @author: AR_DigitalTwin
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .Sampler import Sampler
from ..utils import InputData, InputTypes
# Internal Modules End------------------------------------------------------------------------------

class Saltelli(Sampler):
  """
    Saltelli cross-sampler for variance-based (Sobol') sensitivity analysis.

    Builds two independent low-discrepancy base samples A and B of size N in the
    D-dimensional unit hypercube (Sobol' sequence via ``scipy.stats.qmc``), then
    forms the D "AB_i" matrices in which column i of A is replaced by column i of
    B (and, optionally, the D "BA_i" matrices for second-order terms). Each unit
    coordinate is mapped to a physical value through the variable's distribution
    inverse CDF (``ppf``), exactly as RAVEN's other distribution-driven samplers
    do, so the sampler composes with any 1-D RAVEN distribution and with the
    native Code interfaces (e.g. Dymola).

    The row order of the emitted design is load-bearing: the ``SobolIndices``
    post-processor reconstructs the A, B, AB_i (and BA_i) blocks positionally,
    so this sampler emits the rows in the canonical SALib-compatible order:
    for each base sample j in [0, N): row A_j, then B_j, then AB_{j,0..D-1}
    (then BA_{j,0..D-1} when second order is on). ``RAVEN_sample_ID`` / ``prefix``
    increment in emission order, so a downstream sort on the sample id restores
    this order even if the runs finish out of order.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Saltelli, cls).getInputSpecification()

    samplerInitInput = InputData.parameterInputFactory("samplerInit",
        descr=r"""collection of top-level settings for the Saltelli cross-sampler.""")
    samplerInitInput.addSub(InputData.parameterInputFactory("initialSeed",
        contentType=InputTypes.IntegerType,
        descr=r"""seed for the Sobol' base-sample generator (reproducible designs).""") )
    samplerInitInput.addSub(InputData.parameterInputFactory("N",
        contentType=InputTypes.IntegerType,
        descr=r"""base sample size. Total model runs are $N(D+2)$ for first/total
              order and $N(2D+2)$ when second-order indices are requested, where
              $D$ is the number of sampled variables. Powers of two are recommended
              for the balance properties of the Sobol' sequence."""))
    samplerInitInput.addSub(InputData.parameterInputFactory("computeSecondOrder",
        contentType=InputTypes.BoolType,
        descr=r"""if True, also emit the BA_i blocks needed for second-order
              (pairwise interaction) Sobol' indices, raising the run count from
              $N(D+2)$ to $N(2D+2)$. Default False."""))
    samplerInitInput.addSub(InputData.parameterInputFactory("scramble",
        contentType=InputTypes.BoolType,
        descr=r"""if True (default), Owen-scramble the Sobol' sequence. Scrambling
              improves the uniformity of the base samples and yields unbiased
              variance estimates when N is not a power of two."""))

    inputSpecification.addSub(samplerInitInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'SAMPLER SALTELLI'
    self.N = None                # int, base sample size
    self.computeSecondOrder = False  # bool, whether to emit BA_i blocks
    self.scramble = True         # bool, whether to scramble the Sobol' sequence
    self.saltelliSeed = None     # int, seed for the base-sample generator
    self._design = None          # np.ndarray (rows, D), unit-hypercube design matrix
    self._varNames = None        # list(str), sampled-variable order matching design columns

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, the xml element node checked against this Sampler's options.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    # reuse the base reader for <variable>/<distribution>/<constant> parsing
    Sampler.readSamplerInit(self, xmlNode)
    init = paramInput.findFirst('samplerInit')
    if init is None:
      self.raiseAnError(IOError, self, f'Saltelli sampler {self.name} needs a <samplerInit> block with at least <N>.')
    nSub = init.findFirst('N')
    if nSub is None:
      self.raiseAnError(IOError, self, f'Saltelli sampler {self.name} needs <N> (base sample size) inside <samplerInit>.')
    self.N = int(nSub.value)
    if self.N < 2:
      self.raiseAnError(IOError, self, f'Saltelli sampler {self.name}: <N> must be >= 2 (got {self.N}).')
    secondSub = init.findFirst('computeSecondOrder')
    if secondSub is not None:
      self.computeSecondOrder = bool(secondSub.value)
    scrSub = init.findFirst('scramble')
    if scrSub is not None:
      self.scramble = bool(scrSub.value)
    seedSub = init.findFirst('initialSeed')
    if seedSub is not None:
      self.saltelliSeed = int(seedSub.value)
    if not self.toBeSampled:
      self.raiseAnError(IOError, self, f'Saltelli sampler {self.name} needs at least one sampled <variable> with a <distribution>.')
    # this sampler supports 1-D distributions only (one column per scalar variable)
    for varName in self.toBeSampled:
      if len(varName.split(',')) > 1:
        self.raiseAnError(IOError, self, f'Saltelli sampler {self.name}: comma-joined (fully correlated) variables '
                          f'are not supported ("{varName}"); declare one <variable> per dimension.')

  def _buildDesign(self):
    """
      Build the Saltelli cross-sample matrix in the unit hypercube. Rows are laid
      out per base sample j: [A_j, B_j, AB_{j,0..D-1}] (+ [BA_{j,0..D-1}] if second
      order). Column order is self._varNames == sorted(self.toBeSampled).
      @ In, None
      @ Out, None
    """
    from scipy.stats import qmc

    self._varNames = sorted(self.toBeSampled)
    D = len(self._varNames)
    N = self.N
    # one Sobol' draw of dimension 2D, split into the two independent base samples
    engine = qmc.Sobol(d=2 * D, scramble=self.scramble, seed=self.saltelliSeed)
    base = engine.random(N)              # (N, 2D)
    A = base[:, :D]                      # (N, D)
    B = base[:, D:]                      # (N, D)
    perRun = (2 * D + 2) if self.computeSecondOrder else (D + 2)
    design = np.empty((N * perRun, D), dtype=float)
    row = 0
    for j in range(N):
      design[row] = A[j]; row += 1                 # A_j
      design[row] = B[j]; row += 1                 # B_j
      for i in range(D):                           # AB_{j,i}: A with column i from B
        rr = A[j].copy(); rr[i] = B[j, i]
        design[row] = rr; row += 1
      if self.computeSecondOrder:
        for i in range(D):                         # BA_{j,i}: B with column i from A
          rr = B[j].copy(); rr[i] = A[j, i]
          design[row] = rr; row += 1
    self._design = design

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. Builds the design
      matrix and sets the run limit.
      @ In, None
      @ Out, None
    """
    self._buildDesign()
    self.limit = int(self._design.shape[0])
    self.raiseAMessage(f'Saltelli design: D={len(self._varNames)}, N={self.N}, '
                       f'second_order={self.computeSecondOrder} -> {self.limit} model runs.')

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take. Maps the current unit-hypercube design row
      to physical values through each variable's inverse CDF and fills self.values.
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model
      @ Out, None
    """
    unitRow = self._design[self.counter - 1]  # counter is incremented (1-based) before this call
    for i, varName in enumerate(self._varNames):
      dist = self.distDict[varName]
      quantile = float(unitRow[i])
      value = dist.ppf(quantile)
      for key in varName.split(','):
        key = key.strip()
        self.values[key] = value
        self.inputInfo['SampledVarsPb'][key] = dist.pdf(value)
        self.inputInfo['ProbabilityWeight-' + key] = 1.0
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight'] = 1.0
    self.inputInfo['SamplerType'] = 'Saltelli'
