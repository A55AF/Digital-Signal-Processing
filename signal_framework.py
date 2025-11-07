import numpy as np


class Signal:
    def __init__(self, signal_type, is_periodic, matrix):
        self.signal_type = signal_type
        self.is_periodic = is_periodic
        self.matrix = matrix

    def split(self):
        if self.signal_type == 0:
            t, a = self.matrix.T
            return t, a
        else:
            a, p = self.matrix.T
            return a, p

    def draw(self, plt, type=0, is_cont=False, label=None, scat=True, frqs=None):
        t = None
        a = None
        p = None
        if self.signal_type == False:
            t, a = self.split()
        else:
            a, p = self.split()

        if is_cont == True:
            if type == 0:
                plt.plot(t, a, label=label)
            elif type == 1:
                plt.plot(frqs, a, label=label)
            elif type == 2:
                plt.plot(frqs, p, label=label)
        else:
            if scat == True:
                if type == 0:
                    plt.scatter(t, a, label=label)
                elif type == 1:
                    plt.scatter(frqs, a, label=label)
                elif type == 2:
                    plt.scatter(frqs, p, label=label)
            else:
                if type == 0:
                    plt.stem(t, a, label=label)
                elif type == 1:
                    plt.stem(frqs, a, label=label)
                elif type == 2:
                    plt.stem(frqs, p, label=label)


# time - amplitude
# freq - amplitude
# freq - phase
