import pylab
import matplotlib.pyplot as plt


def ploting(loss_list_train, loss_list_test):
    pylab.plot(loss_list_train[0], linestyle="-", lw="1", color='y', label='ResNet20')
    pylab.plot(loss_list_train[1], linestyle="-", lw="1", color='c', label='ResNet32')
    pylab.plot(loss_list_train[2], linestyle="-", lw="1", color='g', label='ResNet44')
    pylab.plot(loss_list_train[3], linestyle="-", lw="1", color='r', label='ResNet56')
    pylab.plot(loss_list_train[4], linestyle="-", lw="1", color='k', label='ResNet110')
    pylab.plot(loss_list_test[0], linestyle="-", lw="2",  color='y')
    pylab.plot(loss_list_test[1], linestyle="-", lw="2", color='c')
    pylab.plot(loss_list_test[2], linestyle="-", lw="2", color='g')
    pylab.plot(loss_list_test[3], linestyle="-", lw="2", color='r')
    pylab.plot(loss_list_test[4], linestyle="-", lw="2", color='k')
    pylab.legend (loc='upper right')
    pylab.ylim(0, 20)
    pylab.grid()
    pylab.show()