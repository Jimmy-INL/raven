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
#***************************************
#* Ishigami analytic test ExternalModel *
#***************************************
#
# The Ishigami function is the canonical benchmark for variance-based
# sensitivity analysis: it has known analytic Sobol' indices, so the
# Saltelli sampler + SobolIndices post-processor can be checked against them.
#
#   y = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1),   x_i ~ U(-pi, pi)
#
# with a=7, b=0.1 the analytic indices are
#   S1  = [0.3139, 0.4424, 0.0000]
#   ST  = [0.5576, 0.4424, 0.2437]
#
import numpy as np

A = 7.0
B = 0.1

def run(raven, Input):
  """
    Ishigami test function.
    @ In, raven, object, RAVEN object container (holds sampled x1,x2,x3)
    @ In, Input, dict, variable information from RAVEN
    @ Out, None. Sets raven.y
  """
  x1 = raven.x1
  x2 = raven.x2
  x3 = raven.x3
  raven.y = np.sin(x1) + A * np.sin(x2) ** 2 + B * (x3 ** 4) * np.sin(x1)
