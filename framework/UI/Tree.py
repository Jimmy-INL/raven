#!/usr/bin/env python
import numpy as np

class Node(object):
  def __init__(self, _id, parent=None, level=0, size=1):
    self.id = _id
    if parent is None:
      self.parent = self
    else:
      self.parent = parent
    self.level = level
    self.size = size
    self.children = []

  def addChild(self,_id, level=0, size=1):
    node = Node(_id,self,level,size)
    self.children.append(node)
    return node

  def getNode(self,_id):
    if _id == self.id:
      return self
    else:
      for child in self.children:
        node = child.getNode(_id)
        if node is not None:
          return node
      return None

  def getLeafCount(self, truncationSize=0, truncationLevel=0):
    if len(self.children) == 0:
      return 1

    truncated = True

    for child in self.children:
      if child.level >= truncationLevel and child.size >= truncationSize:
        truncated = False

    count = 0
    for child in self.children:
      if not truncated:
        count += child.getLeafCount(truncationSize, truncationLevel)

    if count == 0:
      return 1

    return count

  def maximumChild(self, truncationSize=0, truncationLevel=0):
    maxChild = None

    truncated = True
    for child in self.children:
      if child.level >= truncationLevel and child.size >= truncationSize:
        truncated = False

    for child in self.children:
      if not truncated:
        if maxChild is None or maxChild.level < child.level:
          maxChild = child

    return maxChild

  def Layout(self,xoffset,width, truncationSize=0, truncationLevel=0):
    ids = [self.id]
    points = [(xoffset+width/2.,self.level)]
    edges = []

    totalCount = self.getLeafCount(truncationSize,truncationLevel)
    if len(self.children) > 0 and totalCount > 1:

      myOffset = xoffset

      def cmp(a,b):
        if a.level > b.level:
          return -1
        return 1

      children = sorted(self.children, cmp=cmp)
      immediateDescendantXs = []
      truncated = True

      for child in children:
        if child.level >= truncationLevel and child.size >= truncationSize:
          truncated = False

      for child in children:
        if not truncated:
          edges.append((self.id,child.id))

          count = child.getLeafCount(truncationSize,truncationLevel)
          myWidth = float(count)/totalCount*width
          (childIds,childPoints,childEdges) = child.Layout(myOffset,myWidth,truncationSize,truncationLevel)
          ids.extend(childIds)
          points.extend(childPoints)
          edges.extend(childEdges)

          if len(childPoints) > 0:
            immediateDescendantXs.append(childPoints[0][0])
          myOffset += myWidth

      ## If this guy has children, then we will readjust its X location to be
      ## the average of its immediate descendants
      if len(immediateDescendantXs) > 0:
        points[0] = (np.average(immediateDescendantXs),self.level)
    return (ids,points,edges)