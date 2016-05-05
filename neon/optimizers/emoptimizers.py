
from math import expm1
from neon.optimizers import ExpSchedule

class TauExpSchedule(ExpSchedule):
    """
    Exponential learning rate schedule.

    Arguments:
        tau (float): exponential decay time constant
    """
    def __init__(self, tau, nepochs):
        # set decay so that true exponential decay and neon "exponential" decay match after nepochs
        super(TauExpSchedule, self).__init__(expm1(nepochs/float(tau))/nepochs)
        self.tau = tau; self.nepochs = nepochs

def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision

class DiscreteTauExpSchedule(TauExpSchedule):
    """
    Discrete Exponential learning rate schedule based using tau with target after nepochs.

    Arguments:
        tau (float): exponential decay time constant
        nepochs (int): number of total epochs to calculate final target rate
        epoch_freq (int): discretization in epochs for calculating rate
    """
    def __init__(self, tau, nepochs, epoch_freq):
        super(DiscreteTauExpSchedule, self).__init__(tau, nepochs)
        self.epoch_freq = int(epoch_freq)

    def get_learning_rate(self, learning_rate, epoch):
        return ExpSchedule.get_learning_rate(self, learning_rate, round_to(epoch,self.epoch_freq))
