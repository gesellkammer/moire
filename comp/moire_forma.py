#!/usr/bin/env python
from __future__ import division

# local
from __future__ import division
from collections import namedtuple 
from em.distribute import partition_fib, partition_fib_maximize_partitions
from em.iterlib import izip, pairwise
from bpf4 import bpf
import numpy
from em.collections import ClassDict
from em import csvtools
from em import iterlib
import ordereddict

"""
A : fade --> trem
B : vocal -- bend -- ...
C : difftones
D : silence
"""

"""
Tree --> Branch --> Iteration --> Node

Tree ------ Branch AAA
                      B
                     AAAAA
                          BB
                        AAAAA
                             BBB
                          AAAA
                             BBBBB
                                  C
                                   
                        

Cada grupo AAA.. es una iteracion

C sucede siempre entre Branches

D sucede cada x branches, donde x es una funcion de t

T = Tree

"""

def parse_durations(s):
    if isinstance(s, str):
        possible_durs = [Dur(*map(int, d.split("/"))) for d in s.split()]
    else:
        possible_durs = map(Dur, s)
    return possible_durs

class Stream:
    registry = {}
    def __init__(self, id, node_possible_values):
        """
        node_possible_values: a seq of possible durations (in pulses), or a string of the form "3/8 5/8 3/4 ..."
        """
        self.id = id
        self.__class__.registry[id] = self
        self.node_possible_durs = parse_durations(node_possible_values) 
        self.node_possible_values = [n.dur for n in self.node_possible_durs]

class SubBranch:
    def __init__(self, kind, index, numit, offsets,
                 pulsecurve, numnodes_minmax, rnd, pointfunc=None):
        """
        kind: A, B, etc
        index: which branch is this
        numit: number of iterations in branch
        offsets: the offset between each iteration
        pulsecurve: a bpf between x:0-1, y:pulsemin, pulsemax
        numnodes_min: the min. num. of nodes in an iteration
        numnodes_max: the max. %
        rnd: randomize the pulse-series
        """
        self.kind = kind
        self.index = index
        S = Stream.registry[kind]
        self.poss_durs = S.node_possible_durs
        self.poss_values = S.node_possible_values
        if kind == 'A':
            duty = A_duties[index]    
        elif kind == 'B':
            duty = 1-A_duties[index]
        self.numit = numit
        self.totaldur = br_durations[index] * duty
        self.pulsecurve = pulsecurve
        self.offsets = offsets
        self.rnd = rnd
        self.numnodes_min, self.numnodes_max = numnodes_minmax
        default_pointfunc = lambda nodedurs: (100*nodedurs[0][0] != nodedurs[0][1])+200*(len(nodedurs[-1]) > len(nodedurs[0]))
        pointfunc = pointfunc or default_pointfunc
        self.pointfuncs = [pointfunc]
    
    @lib.returns_tuple("nodes_per_it pulseseries nodedurs dursecs")
    def get_talea(self, pointfuncs=[]):
        P = []
        offsets = self.offsets
        totaldur = self.totaldur
        possible_configurations = branch_get_possible_configurations(self.numit, self.numnodes_min, self.numnodes_max, offsets)
        for nodes_per_it in possible_configurations:
            num_nodes = sum(nodes_per_it)
            num_unique_nodes = get_num_unique_nodes(nodes_per_it, offsets)
            pulseseries = node_pulse_series(num_unique_nodes, pulsecurve=self.pulsecurve, rndm=self.rnd, 
                                            possible_values=self.poss_values)
            nodeindices, nodedurs = branch_expand(pulseseries, nodes_per_it, offsets)
            dursecs = sum(iterlib.flatten(nodedurs)) * (60/globaltempo)
            diff = abs(totaldur - dursecs)
            extrapoints = 0
            for func in pointfuncs:
                extrapoints += func(nodedurs)
            # points: cuanto max grande, mejor
            points = (1000 - diff) + extrapoints
            P.append([points, diff, num_unique_nodes, nodedurs, nodes_per_it])
        P.sort(key=lambda x:x[0])
        points, diff, unique_nodes, nodedurs, nodes_per_it = P[-1]
        if diff/totaldur > 0.1:
            print "warning: the best solution differs from the desired duration in {percent}%".format(percent=round(diff/A.totaldur*100))
        dursecs = sum(iterlib.flatten(nodedurs))*(60/globaltempo)
        return nodes_per_it, pulseseries, nodedurs, dursecs

