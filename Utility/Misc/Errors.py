class EigenvalueNumberError(Exception):
    def __init__(self, all_vals, unique_vals):
        self.all_vals, self.unique_vals = all_vals, unique_vals


### Related to Floquet Evolution
class PrecisionFloquetEvolutionError(Exception):
    """Thrown when precision is lower than required (precision is defined in the
    Hamiltonian class)"""
    def __init__(self, precision, precisionExp, ntrunc):
        self.precision = precision
        self.precisionExp = precisionExp
        self.ntrunc = ntrunc

class BZNbError(Exception):
    """Thrown when dim(FBZ) != dim(H) """
    def __init__(self, nb, nbExp):
        self.nb = nb
        self.nbExp = nbExp


class TruncMotionalError(Exception):
    """Thrown when truncation of n_mot is too strong """
    def __init__(self, nb_mot):
        self.nb_mot = nb_mot


class TruncFloquetlError(Exception):
    """Thrown when dim(FBZ) != dim(H) """
    def __init__(self, nb_trunc):
        self.nb_trunc = nb_trunc

class NotUnitaryError(Exception):
    """Thrown when Matrix is not unitary / sumproba != 1 """
    def __init__(self, nb_trunc, nb_mot):
        self.nb_trunc = nb_trunc
        self.nb_mot = nb_mot

class TooBigError(Exception):
    """Thrown when the task is supposed to be too big (e.g. diago of a matrix)"""
    def __init__(self, size):
        self.size = size


### GENERAL
class NotHError(Exception):
    """ Thrown when matrix is not Hermitian    """
    def __init__(self, nameMatrix):
        self.nameMatrix = nameMatrix



### Not defined
        
class UsageError(Exception):
    pass

