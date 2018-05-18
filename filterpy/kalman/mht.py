# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:58:16 2018

@author: rlabbe
"""
from copy import deepcopy
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import kinematic_kf

class Node(object):
    _last_id = 1
    def __init__(self, kf, z=None):
        self.kf = kf
        self.score = 1.0
        self.uid = Node._last_id
        Node._last_id += 1
        self.z = z

        self.clear_ref()


    def clear_ref(self):
        """ Empty out parent and children, and set depth to one. """

        self.parent = None  # if None, I'm the root of the tree!
        self.children = {}
        self.depth = 1
        self.score = 1.0


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

        return 'Node {:3d}: parent {:3d} depth {:3d} score {:.2f} z {}'.format(
                self.uid, pid, self.depth, self.score, zstr)


    def copy(self, z=None):
        return Node(deepcopy(self.kf), z)


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



# broken into separate tree and nodes so we can keep a master list
# of nodes and leaves. We need to traverse leaves to add measurements
# not sure if we need master list of nodes, however.

class Tree(object):

    def __init__(self, node=None):
        """ Create a Tree with an optional top node"""

        self.clear()
        if node is not None:
            self.create(node)


    def clear(self):
        """ Delete everything in the tree """

        self.nodes = set()

        # handy reference - keeps all the current leaves so we don't have
        # to search to find them
        self.leaves = {}

        self.head = None


    def is_empty(self):
        """ Returns true if the tree contains no nodes"""

        # check for leaves is strictly not necessary, but ends up being
        # a code consistentcy check as it can only assert if there is a bug
        return len(self.nodes) == 0 and len(self.leaves) == 0


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

        self.nodes.add(node)
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
        self.nodes.add(child)


        if child.is_leaf():
            self.leaves[child.uid] = child

        # parent cannot be a leaf, so remove from leaf list
        if parent in self.leaves:
            self.leaves.remove(parent)



    def delete(self, node):
        # sanity check
        assert node in self.nodes

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
        parent.children.remove(node)

        self.leaves.remove(node)
        # parent may have become a new leaf
        if parent.is_leaf():
            self.leaves.add(parent)

        del self.uids[node.uid]


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
        print()
        print('level {}'.format(children[0].depth))
    for child in children:
        print(child, child.kf.x_post.T)

    print_tree(t, children)


if __name__ == '__main__':
    from filterpy.stats import mahalanobis

    def ptree(tree):
        return sorted(tree.nodes, key=lambda x : x.uid)

    N = 4
    zs = [i + .01*np.random.randn() for i in (range(N))]
    zs2 =[i + 2*np.random.randn() for i in (range(N))]

    t = Tree()
    for z1, z2 in zip(zs, zs2):
        associated = False
        add =[]
        for leaf in t.leaves.values():
            print(type(leaf))
            leaf.kf.predict()
            d = mahalanobis(z1, leaf.kf.x[0], leaf.kf.P[0,0])
            print('maha', d)
            if d < 3.: #std
                associated = True
                child = leaf.copy(z1)
                child.kf.update(z1)
                child.score = np.exp(-child.kf.mahalanobis)
                add.append((leaf, child))
                print('adding leaf', child.uid, 'to', leaf.uid)

            child = leaf.copy()
            print(child.score)
            add.append((leaf, child))

        if not associated:
            print(z1, 'is trash')
            if len(t) == 0:
                kf = kinematic_kf(1, 1)
                kf.x[0] = z1
                kf.update(z1) # compute reasonable log_likelihood
                t.create(Node(kf))
                print('making a tree with', z1)
        # add no match prediction
        # this makes a

        for c in add:
            t.add_child(*c)

    branch = child.branch()

    for b in branch:
        print(f'{b.kf.z[0,0]:.4f}, {b.kf.x_post.T}, {b.kf.log_likelihood:.4f}, {b.kf.mahalanobis:.4f}')

    from pprint import pprint
    pprint(ptree(t))
    print()

    print_tree(t, t.head)



