# -*- coding: utf-8 -*-
"""Copyright 2018 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from copy import deepcopy
import math

class Node(object):
    _last_id = 1
    null_update_probability = 0.13

    def _new_id():
        Node._last_id += 1
        return Node._last_id


    def __init__(self, kf, z=None):
        self.clear_ref()

        self.kf = kf
        self.uid = Node._last_id
        Node._last_id += 1

        self.num_updates = 0
        self.z = z
        self.update(z) # safe even if z is None


    def clear_ref(self):
        """ Empty out parent and children, and set depth to one. """

        self.parent = None  # if None, I'm the root of the tree!
        self.children = {}
        self.depth = 1
        self.score = 0.0
        self.z = None
        self.update_depth = 0 # depth of most recent update


    def update(self, z):
        self.z = z

        if z is not None:
            self.kf.update(z)
            p = math.exp(-self.kf.mahalanobis)
        else:
            p = Node.null_update_probability

        self.score = (self.score * self.num_updates + p) / (self.num_updates + 1)
        self.update_depth = self.depth
        if z  is not None:
            self.num_updates += 1


    def add_child(self, child):
        child.depth = self.depth + 1
        self.children[child.uid] = child


    def is_root(self):
        return self.parent is None


    def is_leaf(self):
        return len(self.children) == 0


    def delete_children(self):
        self.children = {}


    def delete_child(self, uid):
        del self.children[uid]


    def __repr__(self):
        if self.parent is None:
            pid = 0
        else:
            pid = self.parent.uid

        if self.z is None:
            zstr = 'None    '
        else:
            zstr = '{:4f}'.format(self.z)

        return 'Node {:3d}: parent {:3d} # children {:3d} depth {:3d} score {:.4f} z {}'.format(
                self.uid, pid, len(self.children), self.depth, self.score, zstr)


    def copy(self, z=None):

        n = deepcopy(self)
        n.uid = Node._new_id()
        n.parent = self
        n.depth += 1

        n.z = None
        return n


    def branch(self):
        """
        Generate a list of all of the nodes up to the root,
        ordered from the head to this node

        node_id will typically be a leaf node, but it doesn't have to be.
        """

        nodes = [self]
        node = self.parent
        while node is not None:
            nodes.append(node)
            node = node.parent
        return list(reversed(nodes))
