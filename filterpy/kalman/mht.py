# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:58:16 2018

@author: rlabbe
"""

import copy


class Node(object):
    def __init__(self, kf):

        self.kf = copy.deepcopy(kf)
        self.parent = None  # if None, I'm the root of the tree!
        self.children = set()
        self.depth = 1


    def is_root(self):
        return self.parent is None


    def is_leaf(self):
        return len(self.children) == 0


    '''def depth(self):
        count = 0
        p = self
        while p is not None:
            p = p.parent
            count += 1
        return count'''

    def __repr__(self):
        return 'Node: ' + hex(id(self)) + ' ' + str(self.kf)


class Tree(object):

    def __init__(self):
        self.clear()


    def clear(self):
        """ Delete everything in the tree """

        self.nodes = set()

        # handy reference - keeps all the leaves so we don't have
        # to search
        self.leaves = set()
        self.head = None


    def is_empty(self):
        """ Returns true if the tree contains no nodes"""

        # check for leaves is strictly not necessary, but ends up being
        # a code consistentcy check as it can only assert if there is a bug
        return len(self.nodes) == 0 and len(self.leaves) == 0


    def create(self, node):
        self.clear()
        self.nodes.add(node)
        self.leaves.add(node)
        self.head = node


    def add_child(self, parent, child):

        assert child not in self.nodes
        assert child not in self.leaves
        assert child.parent is None

        assert len(child.children) == 0
        assert parent is not None

        child.parent = parent

        # add to parent
        parent.children.add(child)

        # add to nodes for easy look up
        self.nodes.add(child)

        if child.is_leaf():
            self.leaves.add(child)

        # parent cannot be a leaf, so remove from leaf list
        if parent in self.leaves:
            self.leaves.remove(parent)




    def delete(self, node):
        print('delete', node)

        # sanity check
        assert node in self.nodes

        if node not in self.nodes:
            return False

        # if I am the root, just delete everything!
        if node.is_root():
            assert node is self.head
            self.clear()
            return


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



    def branch(self, node):
        """
        given a node id, generate a list of all of the kf up to the root,
        ordered from the head to the specified node

        node_id will typically be a leaf node, but it doesn't have to be.
        """

        nodes = [node]

        node = node.parent
        while node is not None:
            nodes.append(node)
            node = node.parent

        return list(reversed(nodes))


    def __len__(self):
        return len(self.nodes)



# for now this is 1 tree
class MultipleHypothesisTracker(object):

    def __init__(self):
        self.id = 0
        self.head = None

        self.tree = dict()


    def create(self, kf):
        pass



    def predict():
        pass





if __name__ == '__main__':

    '''from filterpy.common import kinematic_kf

    mht = MultipleHypothesisTracker()


    kf = kinematic_kf(dim=1, order=1, dt=1, dim_z=1)

    mht.create(kf)'''

    t = Tree()
    n = Node(1)

    t.create(n)

    n2 = Node(2)
    t.add_child(n, n2)

    assert len(t.branch(n2)) == 2
    assert len(t.branch(n)) == 1

    assert n.parent == None
    assert n.is_root()
    assert not n.is_leaf()
    assert not n2.is_root()
    assert n2.is_leaf()

    assert n.depth() == 1
    assert n2.depth() == 2












