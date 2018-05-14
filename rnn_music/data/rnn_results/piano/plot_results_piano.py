# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 14:09:57 2015

@author: Zhe Gan
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

""" Piano dataset. """

data = np.load('piano_sgd.npz')
sgd_train_negll = data['train_negll']
sgd_valid_negll = data['valid_negll']
sgd_test_negll = data['test_negll']
sgd_history_negll = data['history_negll']

data = np.load('piano_momentum.npz')
momentum_train_negll = data['train_negll']
momentum_valid_negll = data['valid_negll']
momentum_test_negll = data['test_negll']
momentum_history_negll = data['history_negll']

data = np.load('piano_RMSprop.npz')
rmsprop_train_negll = data['train_negll']
rmsprop_valid_negll = data['valid_negll']
rmsprop_test_negll = data['test_negll']
rmsprop_history_negll = data['history_negll']

data = np.load('piano_adam.npz')
adam_train_negll = data['train_negll']
adam_valid_negll = data['valid_negll']
adam_test_negll = data['test_negll']
adam_history_negll = data['history_negll']

data = np.load('piano_santa_explore_refine.npz')
santa_train_negll = data['train_negll']
santa_valid_negll = data['valid_negll']
santa_test_negll = data['test_negll']
santa_history_negll = data['history_negll']

data = np.load('piano_santa_small_learn_rate.npz')
santa_s_train_negll = data['train_negll']
santa_s_valid_negll = data['valid_negll']
santa_s_test_negll = data['test_negll']
santa_s_history_negll = data['history_negll']

fig = plt.figure()
adjustFigAspect(fig,aspect=1.2)
ax = fig.add_subplot(111)
plt.plot(sgd_history_negll[:,2],'c', label='SGD',linewidth=2.0, alpha=1.)
plt.plot(momentum_history_negll[:,2],'b', label='SGD-M',linewidth=2.0, alpha=1.)
plt.plot(rmsprop_history_negll[:,2],'r', label='RMSprop',linewidth=2.0, alpha=1.)
plt.plot(adam_history_negll[:,2],'g', label='Adam',linewidth=2.0, alpha=1.)
plt.plot(santa_history_negll[:,2],'k', label='Santa',linewidth=2.0, alpha=1.)
plt.plot(santa_s_history_negll[:,2],'m', label='Santa-s',linewidth=2.0, alpha=1.)
plt.legend(prop={'size':12})
plt.ylim((6,14))
plt.xlim((0,80))
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Negative Log-likelihood',fontsize=12)
plt.title('Piano train',fontsize=12)
plt.savefig('piano_train_with_title.pdf',bbox_inches = 'tight')


fig = plt.figure()
adjustFigAspect(fig,aspect=1.2)
ax = fig.add_subplot(111)
plt.plot(sgd_history_negll[:,0],'c', label='SGD',linewidth=2.0, alpha=1.)
plt.plot(momentum_history_negll[:,0],'b', label='SGD-M',linewidth=2.0, alpha=1.)
plt.plot(rmsprop_history_negll[:,0],'r', label='RMSprop',linewidth=2.0, alpha=1.)
plt.plot(adam_history_negll[:,0],'g', label='Adam',linewidth=2.0, alpha=1.)
plt.plot(santa_history_negll[:,0],'k', label='Santa',linewidth=2.0, alpha=1.)
plt.plot(santa_s_history_negll[:,0],'m', label='Santa-s',linewidth=2.0, alpha=1.)
#plt.legend(prop={'size':12})
plt.ylim((6,14))
plt.xlim((0,80))
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Negative Log-likelihood',fontsize=12)
plt.title('Piano valid',fontsize=12)
plt.savefig('piano_valid_with_title.pdf',bbox_inches = 'tight')


fig = plt.figure()
adjustFigAspect(fig,aspect=1.2)
ax = fig.add_subplot(111)
plt.plot(sgd_history_negll[:,1],'c', label='SGD',linewidth=2.0, alpha=1.)
plt.plot(momentum_history_negll[:,1],'b', label='SGD-M',linewidth=2.0, alpha=1.)
plt.plot(rmsprop_history_negll[:,1],'r', label='RMSprop',linewidth=2.0, alpha=1.)
plt.plot(adam_history_negll[:,1],'g', label='Adam',linewidth=2.0, alpha=1.)
plt.plot(santa_history_negll[:,1],'k', label='Santa',linewidth=2.0, alpha=1.)
plt.plot(santa_s_history_negll[:,1],'m', label='Santa-s',linewidth=2.0, alpha=1.)
#plt.legend(prop={'size':12})
plt.ylim((6,14))
plt.xlim((0,80))
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Negative Log-likelihood',fontsize=12)
plt.title('Piano test',fontsize=12)
plt.savefig('piano_test_with_title.pdf',bbox_inches = 'tight')

