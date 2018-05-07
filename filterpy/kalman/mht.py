# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:58:16 2018

@author: rlabbe
"""



class Node(object):
    def __init__(self, kf):

        self.kf = kf
        self.score = 1.0

        self.clear_ref()


    def clear_ref(self):
        """ Empty out parent and children, and set depth to one. """

        self.parent = None  # if None, I'm the root of the tree!
        self.children = set()
        self.depth = 1
        self.score = 1.0

    def is_root(self):
        return self.parent is None


    def is_leaf(self):
        return len(self.children) == 0


    def delete_children(self):
        self.children = set()


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


    def __repr__(self):
        return 'Node: ' + hex(id(self)) + ' ' + str(self.kf)


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
        child.depth = parent.depth + 1

        # add to nodes for easy look up
        self.nodes.add(child)

        if child.is_leaf():
            self.leaves.add(child)

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

    assert len(n2.branch()) == 2
    assert len(n.branch()) == 1

    assert n.parent == None
    assert n.is_root()
    assert not n.is_leaf()
    assert not n2.is_root()
    assert n2.is_leaf()

    assert n.depth == 1
    assert n2.depth == 2












