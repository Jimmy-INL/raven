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
#* Trivial linear ExternalModel        *
#***************************************
#
# A deterministic linear response used to exercise the Saltelli sampler's
# A / B / AB_i design construction. The point of this test is the *design*
# (row count and the sampled input values), not the sensitivity math, so the
# model is kept as simple as possible: y = x1 + 2*x2.
#
def run(raven, Input):
  """
    Linear test function.
    @ In, raven, object, RAVEN object container (holds sampled x1,x2)
    @ In, Input, dict, variable information from RAVEN
    @ Out, None. Sets raven.y
  """
  raven.y = raven.x1 + 2.0 * raven.x2
