# -*- coding: utf-8 -*-
"""
Created on Fri May 18 06:21:31 2018

@author: roger
"""

from filterpy.stats import mahalanobis

from filterpy.common import kinematic_kf
from filterpy.kalman.mht import Tree, Node


def test_tree():
    t = Tree()
    n = Node(kinematic_kf(1, 1))

    t.create(n)

    n2 = Node(kinematic_kf(1, 1))
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


if __name__ == '__main__':
    test_tree()
