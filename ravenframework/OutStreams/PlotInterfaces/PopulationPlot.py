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
Created on November 20th, 2021

@author: mandd
"""

# External Imports
import matplotlib.pyplot as plt
import numpy as np

# Internal Imports
from ...utils import plotUtils
from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class PopulationPlot(PlotInterface):
  """
    Plots population coordinate in input and output space
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.setStrictMode(False)
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the data should be taken for this plotter.
              This should be the SolutionExport for a MultiRun with an Optimizer."""))
    spec.addSub(InputData.parameterInputFactory('vars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject whose optimization paths should be plotted."""))
    spec.addSub(InputData.parameterInputFactory('logVars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject to be plotted on a log scale."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Names of the variable that refers to the batch index"""))
    spec.addSub(InputData.parameterInputFactory('how', contentType=InputTypes.StringType,
        descr=r"""Digital format of the generated picture"""))
    spec.addSub(InputData.parameterInputFactory('summary', contentType=InputTypes.BoolType,
        descr=r"""If True, plot min/mean/max lines instead of shaded min-max band."""))
    spec.addSub(InputData.parameterInputFactory('feasibleOnlyBand', contentType=InputTypes.BoolType,
        descr=r"""If True, overlay a min-max band computed only from feasible points (constraints >= 0)."""))
    spec.addSub(InputData.parameterInputFactory('violationOnlyBand', contentType=InputTypes.BoolType,
        descr=r"""If True, overlay a min-max band computed only from violating points (constraints < 0)."""))
    spec.addSub(InputData.parameterInputFactory('hideOverallBand', contentType=InputTypes.BoolType,
        descr=r"""If True, do not draw the overall population band when plotting constraint overlays."""))
    spec.addSub(InputData.parameterInputFactory('constraintVars', contentType=InputTypes.StringListType,
        descr=r"""Constraint evaluation variables used to determine feasibility."""))
    spec.addSub(InputData.parameterInputFactory('constraintColors', contentType=InputTypes.StringListType,
        descr=r"""Optional list of colors to use for each constraint overlay band. Must match constraintVars length."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag   = 'GAPopulation Plot'
    self.source     = None      # reference to DataObject source
    self.sourceName = None      # name of DataObject source
    self.vars       = None      # variables to plot
    self.logVars    = None      # variables to plot in log scale
    self.index      = None      # index ID for each batch
    self.how        = None      # format of the generated picture
    self.summary    = False     # plot min/mean/max lines instead of band
    self.feasibleOnlyBand = False
    self.violationOnlyBand = False
    self.hideOverallBand = False
    self.constraintVars = []
    self.constraintColors = []

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    params, notFound = spec.findNodesAndExtractValues(['source','vars','index','how'])

    for node in notFound:
      self.raiseAnError(IOError, "Missing " +str(node) +" node in the PopulationPlot " + str(self.name))
    else:
      self.sourceName = params['source']
      self.vars       = params['vars']
      self.index      = params['index']
      self.how        = params['how']

    params, notFound = spec.findNodesAndExtractValues(['logVars'])
    if notFound:
      self.logVars = []
    else:
      self.logVars = params['logVars']

    params, notFound = spec.findNodesAndExtractValues(['summary'])
    if notFound:
      self.summary = False
    else:
      self.summary = bool(params['summary'])

    params, notFound = spec.findNodesAndExtractValues(['feasibleOnlyBand'])
    if notFound:
      self.feasibleOnlyBand = False
    else:
      self.feasibleOnlyBand = bool(params['feasibleOnlyBand'])
    params, notFound = spec.findNodesAndExtractValues(['violationOnlyBand'])
    if notFound:
      self.violationOnlyBand = False
    else:
      self.violationOnlyBand = bool(params['violationOnlyBand'])

    params, notFound = spec.findNodesAndExtractValues(['hideOverallBand'])
    if notFound:
      self.hideOverallBand = False
    else:
      self.hideOverallBand = bool(params['hideOverallBand'])

    params, notFound = spec.findNodesAndExtractValues(['constraintVars'])
    if notFound:
      self.constraintVars = []
    else:
      self.constraintVars = params['constraintVars']

    params, notFound = spec.findNodesAndExtractValues(['constraintColors'])
    if notFound:
      self.constraintColors = []
    else:
      self.constraintColors = params['constraintColors']


  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
                                current step. The sources are searched into this.
      @ Out, None
    """
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" was found in the Step for SamplePlot "{self.name}"!')
    self.source = src

    dataVars = self.source.getVars()
    missing = [var for var in (self.vars) if var not in dataVars]
    if missing:
      msg = f'Source DataObject "{self.source.name}" is missing the following variables ' +\
            f'expected by OptPath plotter "{self.name}": '
      msg += ', '.join(f'"{m}"' for m in missing)
      self.raiseAnError(IOError, msg)

  def run(self):
    """
      Main run method.
      @ In, None
      @ Out, None
    """
    data = self.source.asDataset().to_dataframe()
    inVars = self.source.getVars(subset='input')

    if self.feasibleOnlyBand and self.violationOnlyBand:
      self.raiseAnError(IOError, "Only one of feasibleOnlyBand or violationOnlyBand can be True.")

    if (self.feasibleOnlyBand or self.violationOnlyBand) and not self.constraintVars:
      self.constraintVars = [col for col in data.columns if col.startswith('ConstraintEvaluation_')]
    if (self.feasibleOnlyBand or self.violationOnlyBand) and not self.constraintVars:
      self.raiseAnError(IOError, "Missing constraint variables for feasible-only band; add <constraintVars> or include ConstraintEvaluation_* in the source.")
    if (self.feasibleOnlyBand or self.violationOnlyBand) and self.constraintColors:
      if len(self.constraintColors) != len(self.constraintVars):
        self.raiseAnError(IOError, f'constraintColors length ({len(self.constraintColors)}) must match constraintVars length ({len(self.constraintVars)}).')
    if (self.feasibleOnlyBand or self.violationOnlyBand) and not self.constraintColors:
      cmap = plt.get_cmap('tab10')
      self.constraintColors = [cmap(i % 10) for i in range(len(self.constraintVars))]

    nFigures = len(self.vars)
    fig, axs = plt.subplots(nFigures,1, figsize=(8, 15))

    minGen = int(min(data[self.index]))
    maxGen = int(max(data[self.index]))

    for indexVar,var in enumerate(self.vars):
      minFit = np.zeros(maxGen-minGen+1)
      maxFit = np.zeros(maxGen-minGen+1)
      avgFit = np.zeros(maxGen-minGen+1)

      feasibleMin = []
      feasibleMax = []
      feasibleAvg = []
      if self.feasibleOnlyBand or self.violationOnlyBand:
        for _ in self.constraintVars:
          feasibleMin.append(np.full(maxGen - minGen + 1, np.nan))
          feasibleMax.append(np.full(maxGen - minGen + 1, np.nan))
          feasibleAvg.append(np.full(maxGen - minGen + 1, np.nan))
      for idx,genID in enumerate(range(minGen,maxGen+1,1)):
        population = data[data[self.index]==genID]
        minFit[idx] = min(population[var])
        maxFit[idx] = max(population[var])
        avgFit[idx] = population[var].mean()
        if self.feasibleOnlyBand or self.violationOnlyBand:
          for c_idx, cvar in enumerate(self.constraintVars):
            if cvar not in population.columns:
              self.raiseAnError(IOError, f'Missing constraint variable \"{cvar}\" in PopulationPlot source.')
            if self.feasibleOnlyBand:
              subset = population[population[cvar] >= 0.0]
            else:
              subset = population[population[cvar] < 0.0]
            if not subset.empty:
              feasibleMin[c_idx][idx] = subset[var].min()
              feasibleMax[c_idx][idx] = subset[var].max()
              feasibleAvg[c_idx][idx] = subset[var].mean()

      xvals = range(minGen, maxGen + 1, 1)
      if self.summary:
        color = 'g' if var in inVars else 'b'
        axs[indexVar].plot(xvals, avgFit, color=color, label='mean')
        axs[indexVar].plot(xvals, minFit, color=color, linestyle='--', label='min')
        axs[indexVar].plot(xvals, maxFit, color=color, linestyle='-.', label='max')
        if var in self.logVars:
          axs[indexVar].set_yscale('log')
        if indexVar == 0:
          axs[indexVar].legend(loc='best')
      else:
        baseColor = 'g' if var in inVars else 'b'
        if not (self.hideOverallBand and (self.feasibleOnlyBand or self.violationOnlyBand)):
          if var in self.logVars:
            plotUtils.errorFill(xvals, avgFit, [minFit,maxFit], color=baseColor, ax=axs[indexVar],logScale=True)
          else:
            plotUtils.errorFill(xvals, avgFit, [minFit,maxFit], color=baseColor, ax=axs[indexVar])
        else:
          axs[indexVar].plot(xvals, minFit, color=baseColor, linestyle='--', label='min')
          axs[indexVar].plot(xvals, maxFit, color=baseColor, linestyle='-.', label='max')
          axs[indexVar].plot(xvals, avgFit, color=baseColor, label='mean')
          if var in self.logVars:
            axs[indexVar].set_yscale('log')
          if indexVar == 0:
            axs[indexVar].legend(loc='best')
        if self.feasibleOnlyBand or self.violationOnlyBand:
          for c_idx, cvar in enumerate(self.constraintVars):
            color = self.constraintColors[c_idx]
            label = cvar.replace('ConstraintEvaluation_', '')
            if var in self.logVars:
              plotUtils.errorFill(xvals, feasibleAvg[c_idx], [feasibleMin[c_idx], feasibleMax[c_idx]],
                                  color=color, alphaFill=0.2, ax=axs[indexVar],logScale=True)
            else:
              plotUtils.errorFill(xvals, feasibleAvg[c_idx], [feasibleMin[c_idx], feasibleMax[c_idx]],
                                  color=color, alphaFill=0.2, ax=axs[indexVar])
            if indexVar == 0:
              axs[indexVar].plot([], [], color=color, label=label)
          if indexVar == 0:
            axs[indexVar].legend(loc='best')
      axs[indexVar].set_ylabel(var)
      if var == self.vars[-1]:
        axs[indexVar].set_xlabel('Batch #')

    fig.tight_layout()

    if self.how in ['png','pdf','svg','jpeg']:
      # create filename
      filename = self._createFilename(defaultName=self.name +'.%s'  % self.how)
      plt.savefig(filename, format=self.how)
    else:
      self.raiseAnError(IOError, f'Digital format of the plot "{self.name}" is not available!')
