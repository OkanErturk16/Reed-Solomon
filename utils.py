import numpy as np
import numba

@numba.jit(nopython = True)
def dec2bin(x,L,order = 'reversed'):
    #----decimal to binary in a reversed order fasion
    #x: integer number to be casted to a binary bit array
    #L: number of bit representing "x"
    #order: MSB or LSB first. if "natural", x_bin[0] is MSB. if "reversed", LSB.
    x_bin = np.zeros(L,dtype=np.uint64)
    divider  = 1<<(L-1)
    x_rem = x
    if order == 'natural':
        for ll in range(L):
            if x_rem >= divider:
                x_bin[L-ll-1] = 1
                x_rem -= divider
            divider = divider>>1

    if order == 'reversed':
        for ll in range(L):
            if x_rem >= divider:
                x_bin[ll] = 1
                x_rem -= divider
            divider = divider>>1
    return x_bin


@numba.jit(nopython = True)
def d2b_rev(x,L):
    """
        Faster version of "dec2bin(x,L order='reversed')"
    """
    x_bin   = np.zeros(L,dtype=np.uint64)
    divider = 1<<(L-1)
    x_rem = x
    for ll in range(L):
        if x_rem >= divider:
            x_bin[ll] = 1
            x_rem -= divider
        divider = divider>>1
    return x_bin


@numba.jit(nopython = True)
def d2b_nat(x,L):
    """
        Faster version of "dec2bin(x,L order='natural')"
    """
    x_bin   = np.zeros(L,dtype=np.uint64)
    divider = 1<<(L-1)
    x_rem = x
    if order == 'natural':
        for ll in range(L):
            if x_rem >= divider:
                x_bin[L-ll-1] = 1
                x_rem -= divider
            divider = divider>>1
    return x_bin



@numba.jit(nopython = True)
def d2b_rev_array(x,L):
    """
        array version of "d2b_rev(x,L order='reversed')"
        x: nd.array
    """

    x_bin = np.zeros((len(x), L),dtype=np.int64)
    
    for ii in range(len(x)):
        divider = 1<<(L-1)
        x_rem = x[ii]
        for ll in range(L):
            if x_rem >= divider:
                x_bin[ii,ll] = 1
                x_rem -= divider
            divider = divider>>1
    return x_bin
