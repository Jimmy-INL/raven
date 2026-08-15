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
  Sobol' sensitivity-index post-processor.

  Consumes the raw model outputs produced by the ``Saltelli`` sampler's
  A / B / AB_i (and optional BA_i) cross-sample design and computes the
  variance-based first-order (``S1``) and total-order (``ST``) Sobol' indices,
  and optionally the pairwise second-order indices (``S2``). The estimators are
  the widely used Saltelli-2010 first-order estimator and the Jansen-1999
  total-order estimator; bootstrap resampling over base samples yields the
  confidence-interval half-widths (``S1_conf`` / ``ST_conf``). Implementation is
  numpy-only (no SALib runtime dependency), so it runs in the stock RAVEN
  environment.

  This is the RAVEN-native counterpart to the "drive with RAVEN, analyze with
  SALib" external route: pairing ``Saltelli`` + ``SobolIndices`` produces the
  same classical indices inside a single RAVEN run.

  Created on 2026-08-14
  @author: AR_DigitalTwin
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes
# Internal Modules End-----------------------------------------------------------

class SobolIndices(PostProcessorInterface):
  """
    Computes Sobol' first/total (and optional second) order sensitivity indices
    from a Saltelli A/B/AB_i sampled PointSet.
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
    inSpec = super(SobolIndices, cls).getInputSpecification()
    inSpec.addSub(InputData.parameterInputFactory('features',
        contentType=InputTypes.StringListType,
        descr=r"""the sampled input variables (the Saltelli columns), in the SAME
              order the Saltelli sampler emitted them (i.e. sorted variable names).
              Determines $D$ and the AB_i block ordering."""))
    inSpec.addSub(InputData.parameterInputFactory('targets',
        contentType=InputTypes.StringListType,
        descr=r"""the scalar output(s) (QoIs) for which to compute Sobol' indices."""))
    inSpec.addSub(InputData.parameterInputFactory('N',
        contentType=InputTypes.IntegerType,
        descr=r"""the base sample size used by the Saltelli sampler. The number of
              rows in the input PointSet must equal $N(D+2)$ (or $N(2D+2)$ with
              second order); this is asserted before estimation."""))
    inSpec.addSub(InputData.parameterInputFactory('computeSecondOrder',
        contentType=InputTypes.BoolType,
        descr=r"""set True only if the Saltelli sampler was run with
              computeSecondOrder=True (so the input has $N(2D+2)$ rows and the
              BA_i blocks are present). Enables the S2 pairwise indices. Default False."""))
    inSpec.addSub(InputData.parameterInputFactory('numBootstrap',
        contentType=InputTypes.IntegerType,
        descr=r"""number of bootstrap resamples (over base samples) used for the
              confidence-interval half-widths. Default 100."""))
    inSpec.addSub(InputData.parameterInputFactory('sampleIDName',
        contentType=InputTypes.StringType,
        descr=r"""name of the monotonically increasing sample-id variable used to
              restore the Saltelli emission order (default 'prefix'). Rows are
              sorted numerically on it before the A/B/AB blocks are reconstructed."""))
    inSpec.addSub(InputData.parameterInputFactory('confidenceLevel',
        contentType=InputTypes.FloatType,
        descr=r"""two-sided confidence level for the bootstrap interval half-widths
              (default 0.95)."""))
    return inSpec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR SOBOL INDICES'
    self.dynamic = False
    self.validDataType = ['PointSet']
    self.outputMultipleRealizations = False
    self.features = None            # list(str), sampled variables (Saltelli columns), sorted order
    self.targets = None             # list(str), QoI outputs
    self.N = None                   # int, base sample size
    self.computeSecondOrder = False # bool
    self.numBootstrap = 100         # int
    self.sampleIDName = 'prefix'    # str
    self.confidenceLevel = 0.95     # float

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      tag = child.getName()
      if tag == 'features':
        self.features = child.value
      elif tag == 'targets':
        self.targets = child.value
      elif tag == 'N':
        self.N = int(child.value)
      elif tag == 'computeSecondOrder':
        self.computeSecondOrder = bool(child.value)
      elif tag == 'numBootstrap':
        self.numBootstrap = int(child.value)
      elif tag == 'sampleIDName':
        self.sampleIDName = child.value
      elif tag == 'confidenceLevel':
        self.confidenceLevel = float(child.value)
    if not self.features:
      self.raiseAnError(IOError, self.printTag, 'requires a <features> list.')
    if not self.targets:
      self.raiseAnError(IOError, self.printTag, 'requires a <targets> list.')
    if self.N is None:
      self.raiseAnError(IOError, self.printTag, 'requires <N> (Saltelli base sample size).')

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format understandable
      by this pp. Returns the single input DataObject.
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject, input data
    """
    if isinstance(currentInp, list):
      if len(currentInp) != 1:
        self.raiseAnError(IOError, self.printTag, f'expects exactly 1 input DataObject, got {len(currentInp)}.')
      currentInp = currentInp[0]
    if currentInp.type not in self.validDataType:
      self.raiseAnError(IOError, self.printTag, f'requires a PointSet input, got "{currentInp.type}".')
    return currentInp

  def _orderedMatrix(self, dataObj):
    """
      Extract, order (by sampleIDName), and validate the sampled feature/target
      matrix from the input PointSet.
      @ In, dataObj, DataObject.PointSet, the Saltelli-sampled data
      @ Out, (featMat, targMat), tuple(np.ndarray, np.ndarray), shapes (rows,D) and (rows,T)
    """
    ds = dataObj.asDataset()
    available = set(dataObj.getVars())
    missingF = [f for f in self.features if f not in available]
    missingT = [t for t in self.targets if t not in available]
    if missingF:
      self.raiseAnError(KeyError, self.printTag, f'features not in input data: {missingF}')
    if missingT:
      self.raiseAnError(KeyError, self.printTag, f'targets not in input data: {missingT}')
    # restore Saltelli emission order
    if self.sampleIDName in available or self.sampleIDName in ds.coords:
      try:
        orderKey = np.asarray(ds[self.sampleIDName].values, dtype=float)
      except Exception:
        orderKey = np.asarray(ds[self.sampleIDName].values)
      order = np.argsort(orderKey, kind='stable')
    else:
      self.raiseAWarning(f'{self.printTag}: sampleID "{self.sampleIDName}" not found; '
                         'assuming rows are already in Saltelli emission order.')
      order = None
    featMat = np.column_stack([np.asarray(ds[f].values, dtype=float).ravel() for f in self.features])
    targMat = np.column_stack([np.asarray(ds[t].values, dtype=float).ravel() for t in self.targets])
    if order is not None:
      featMat = featMat[order]
      targMat = targMat[order]
    D = len(self.features)
    perRun = (2 * D + 2) if self.computeSecondOrder else (D + 2)
    expected = self.N * perRun
    if featMat.shape[0] != expected:
      self.raiseAnError(IOError, self.printTag,
        f'input has {featMat.shape[0]} rows but the Saltelli design expects {expected} '
        f'(N={self.N}, D={D}, computeSecondOrder={self.computeSecondOrder}). '
        'Refusing to estimate indices from a mis-sized matrix.')
    return featMat, targMat

  def _splitBlocks(self, y):
    """
      Split one Saltelli-ordered output vector into A, B, AB (and BA) blocks.
      Emission order per base sample j is [A_j, B_j, AB_{j,0..D-1}(, BA_{j,0..D-1})].
      @ In, y, np.ndarray, shape (N*perRun,), one target's outputs in emission order
      @ Out, (A, B, AB, BA), tuple of np.ndarrays: A,B shape (N,); AB,BA shape (N,D)
    """
    D = len(self.features)
    perRun = (2 * D + 2) if self.computeSecondOrder else (D + 2)
    N = self.N
    Y = y.reshape(N, perRun)
    A = Y[:, 0]
    B = Y[:, 1]
    AB = Y[:, 2:2 + D]
    BA = Y[:, 2 + D:2 + 2 * D] if self.computeSecondOrder else None
    return A, B, AB, BA

  @staticmethod
  def _firstOrder(A, B, AB_i):
    """Saltelli-2010 first-order estimator: S_i = mean(B*(AB_i - A)) / Var(all)."""
    var = np.var(np.concatenate([A, B]), ddof=1)
    if var <= 0:
      return 0.0
    return float(np.mean(B * (AB_i - A)) / var)

  @staticmethod
  def _totalOrder(A, B, AB_i):
    """Jansen-1999 total-order estimator: S_Ti = mean((A - AB_i)^2) / (2 Var(all))."""
    var = np.var(np.concatenate([A, B]), ddof=1)
    if var <= 0:
      return 0.0
    return float(np.mean((A - AB_i) ** 2) / (2.0 * var))

  @staticmethod
  def _secondOrder(A, B, AB_i, AB_j, BA_i, Si, Sj):
    """Saltelli second-order estimator for the closed pair minus the two first orders."""
    var = np.var(np.concatenate([A, B]), ddof=1)
    if var <= 0:
      return 0.0
    Vij = np.mean(BA_i * AB_j - A * B) / var
    return float(Vij - Si - Sj)

  def _bootstrapConf(self, A, B, AB, estimator, pointEst):
    """
      Bootstrap CI half-widths over base samples for a per-feature estimator.
      @ In, A, B, np.ndarray (N,); AB, np.ndarray (N,D)
      @ In, estimator, callable(A,B,AB_i)->float
      @ In, pointEst, np.ndarray (D,), the full-sample point estimates
      @ Out, conf, np.ndarray (D,), CI half-widths
    """
    D = AB.shape[1]
    N = A.shape[0]
    nb = self.numBootstrap
    if nb <= 1:
      return np.zeros(D)
    rng = np.random.default_rng(0)
    boot = np.empty((nb, D))
    for b in range(nb):
      idx = rng.integers(0, N, N)
      for i in range(D):
        boot[b, i] = estimator(A[idx], B[idx], AB[idx, i])
    alpha = 1.0 - self.confidenceLevel
    lo = np.percentile(boot, 100 * alpha / 2.0, axis=0)
    hi = np.percentile(boot, 100 * (1.0 - alpha / 2.0), axis=0)
    return (hi - lo) / 2.0

  def run(self, inputs):
    """
      This method executes the postprocessor action.
      @ In, inputs, list(object), objects containing the data to process.
      @ Out, outputDict, dict, {varName: np.atleast_1d(value)} of Sobol' indices.
    """
    dataObj = self.inputToInternal(inputs)
    featMat, targMat = self._orderedMatrix(dataObj)
    D = len(self.features)
    outputDict = {}
    for tIdx, target in enumerate(self.targets):
      A, B, AB, BA = self._splitBlocks(targMat[:, tIdx])
      s1 = np.array([self._firstOrder(A, B, AB[:, i]) for i in range(D)])
      st = np.array([self._totalOrder(A, B, AB[:, i]) for i in range(D)])
      s1Conf = self._bootstrapConf(A, B, AB, self._firstOrder, s1)
      stConf = self._bootstrapConf(A, B, AB, self._totalOrder, st)
      for i, feat in enumerate(self.features):
        outputDict['S1_' + target + '_' + feat] = np.atleast_1d(s1[i])
        outputDict['S1_conf_' + target + '_' + feat] = np.atleast_1d(s1Conf[i])
        outputDict['ST_' + target + '_' + feat] = np.atleast_1d(st[i])
        outputDict['ST_conf_' + target + '_' + feat] = np.atleast_1d(stConf[i])
      if self.computeSecondOrder and BA is not None:
        for i in range(D):
          for j in range(i + 1, D):
            s2 = self._secondOrder(A, B, AB[:, i], AB[:, j], BA[:, i], s1[i], s1[j])
            outputDict['S2_' + target + '_' + self.features[i] + '_' + self.features[j]] = np.atleast_1d(s2)
    return outputDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object.
      @ In, finishedJob, JobHandler instance running this post-processor
      @ In, output, dataObjects, the object where computed results are placed
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]
    if output.type in ['PointSet', 'HistorySet']:
      self.raiseADebug('Dumping Sobol indices into data object named ' + output.name)
      output.addRealization(outputDict)
    else:
      self.raiseAnError(IOError, self.printTag, f'output type "{output.type}" not supported (use a PointSet).')
