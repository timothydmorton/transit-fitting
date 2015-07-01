from __future__ import print_function, division

def t_folded(t, per, ep):
    return (t + per/2 - ep) % per - (per/2)
