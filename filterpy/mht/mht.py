# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:58:16 2018

@author: rlabbe
"""
from copy import deepcopy
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import kinematic_kf, Q_discrete_white_noise
from node import Node



# broken into separate tree and nodes so we can keep a master list
# of nodes and leaves. We need to traverse leaves to add measurements
# not sure if we need master list of nodes, however.

class Tree(object):

    """

    Attributes
    ----------

    head : Node, or Node, read_only

        Head of the tree.

    leaves : dict{Node}, read_only
        Stores all nodes that are currently in the bottom of the tree.
        This lets us avoid tree traversal when we add measurements.

        Indexed by Node.uid.

     nodes : dict{Node}, read_only
         Stores every Node in the tree. This is not strictly needed, it is
         just a handy way to get all the nodes without having to traverse
         the tree.

         Indexed by Node.uid
    """

    def __init__(self, node=None):
        """ Create a Tree with an optional top node"""

        self.clear()
        if node is not None:
            self.create(node)


    def clear(self):
        """ Delete everything in the tree """

        self.nodes = {}
        self.leaves = {}
        self.head = None


    def is_empty(self):
        """ Returns true if the tree contains no nodes"""
        return self.head is None


    def create(self, node):
        """
        Creates a tree with the head tree `node`. Will destroy all data
        currently stored in the tree
        """
        self.clear()

        # if thsese asserts are not True then node must belong to another
        # tree, and we can't add it to this one
        assert node.is_leaf() and node.is_root()

        # make sure node is initialized properly
        node.depth = 1
        node.parent = None

        self.nodes[node.uid] = node
        self.leaves[node.uid] = node
        self.head = node


    def add_child(self, parent, child):

        assert child not in self.nodes
        assert child not in self.leaves

        assert len(child.children) == 0
        assert parent is not None

        child.parent = parent

        # add to parent
        parent.add_child(child)

        # add to nodes for easy look up
        self.nodes[child.uid] = child

        if child.is_leaf():
            self.leaves[child.uid] = child

        # parent cannot be a leaf, so remove from leaf list
        if parent.uid in self.leaves:
            del self.leaves[parent.uid]


    def delete(self, node):
        # sanity check
        assert node.uid in self.nodes

        del self.nodes[node.uid]

        if node.uid in self.leaves:
            del self.leaves[node.uid]

        # if I am the root, just delete everything!
        if node.is_root():
            assert node is self.head
            self.clear()
            return

        # recursively delete children; have to do this to ensure they
        # are all removed from self.nodes and self.leaves
        for n in node.children:
            self.delete(n)

        # now node is a leaf, so delete it and remove from parent
        assert node.is_leaf()
        parent = node.parent
        del parent.children[node.uid]

        # parent may have become a new leaf
        if parent.is_leaf():
            self.leaves[parent.uid] = parent

    def highest_score(self):
        if len(t.leaves) == 0:
            return None

        return max(self.leaves.values(), key=lambda leaf: leaf.score)


    def predict(self):
        for leaf in self.leaves.values():
            leaf.predict()


    def __len__(self):
        return len(self.nodes)


def print_tree(t, level):
    if level is None:
        return

    try:
        level[0]
        if len(level) == 0:
            return

    except TypeError:
        print('Level 1')
        print(level)
        level = [level]
    except IndexError:
        return # 0 length list

    children = []
    for node in level:
        leaves = list(node.children.values())
        children.extend(sorted(leaves, key=lambda n : n.uid))

    if len(children) > 0:
        print('------------------------------------------------------------------')
        #print('level {}'.format(children[0].depth))
    for child in children:
        print(child, child.kf.x_post.T)

    print_tree(t, children)


if __name__ == '__main__':
    from pprint import pprint
    from filterpy.stats import mahalanobis

    def ptree(tree):
        return sorted(tree.nodes.values(), key=lambda x : x.uid)

    N = 5
    z_std = 0.01
    zs1 = [i+1 + 5*z_std*np.random.randn() for i in (range(N))]
    zs2 = [i+1 + 2*np.random.randn() for i in (range(N))]

    measurements = list(zip(zs1, zs2))

    t = Tree()
    kf = kinematic_kf(1, 1, dt=1.)
    kf.P *= .05
    kf.Q = Q_discrete_white_noise(2, dt=1., var=0.1)
    kf.R *= z_std**2

    kf.x[0] = 0.
    kf.x[1] = 1.
    t.create(Node(kf, 0))


    print('making a tree with', 0)
    for lvl, zs in enumerate(measurements):
        print()
        print('level', lvl+1)

        add =[]

        t.predict()
        for leaf in t.leaves.values():


            for i, z in enumerate(zs):
                associated = False

                d = mahalanobis(z, leaf.kf.x[0], leaf.kf.P[0,0])
                print(f'z={z:.4f}  maha {d:.3f}')
                if d < 3.: #std
                    associated = True
                    child = leaf.copy()

                    child.update(z)
                    add.append((leaf, child))
                    print(i, 'adding leaf', child.uid, 'to', leaf.uid)
                    assert len(child.children) == 0

            # add no match prediction
            if True or leaf.depth > 1: # never predict on head
                child = leaf.copy()
                child.update(None)
                print('adding prediction', child.uid, 'to leaf', leaf.uid)
                add.append((leaf, child))

            if not associated:
                print(z, 'is trash')

        #print('adding', add)
        for c in add:
            t.add_child(*c)


    print_tree(t, t.head)

    print()
    branch = t.highest_score().branch()
    pprint(branch)
