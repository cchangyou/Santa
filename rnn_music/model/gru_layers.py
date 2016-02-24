
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p
from utils import ortho_weight, uniform_weight, zero_bias

    
""" Decoder using GRU Recurrent Neural Network. """

def param_init_decoder(options, params, prefix='decoder_gru'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix,'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix,'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(2*n_h)

    Wx = uniform_weight(n_x, n_h)
    params[_p(prefix,'Wx')] = Wx
    
    Ux = ortho_weight(n_h)
    params[_p(prefix,'Ux')] = Ux
    
    params[_p(prefix,'bx')] = zero_bias(n_h)
    
    params[_p(prefix,'b0')] = zero_bias(n_h)

    return params   
    

def decoder_layer(tparams, state_below, prefix='decoder_gru'):
    
    """ state_below: size of n_steps *  n_x 
    """

    n_steps = state_below.shape[0]
    n_h = tparams[_p(prefix,'Ux')].shape[1]
        
    state_belowx0 = tparams[_p(prefix, 'b0')]
    h0vec = tensor.tanh(state_belowx0)
    h0 = h0vec.dimshuffle('x',0)
    
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1)*dim]
        
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')])  + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    
    def _step_slice(x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        u = tensor.nnet.sigmoid(_slice(preact, 1, n_h))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h

        return h
    
    seqs = [state_below_[:n_steps-1], state_belowx[:n_steps-1]]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [h0vec],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps-1)
                                
    #h0x = h0.dimshuffle('x',0,1)
                            
    return tensor.concatenate((h0,rval))
