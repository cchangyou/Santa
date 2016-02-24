
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

from gru_layers import param_init_decoder, decoder_layer

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_x = options['n_x']  
    n_h = options['n_h']
    
    params = OrderedDict()
    params = param_init_decoder(options,params)
    
    params['Vhid'] = uniform_weight(n_h,n_x)
    params['bhid'] = zero_bias(n_x)                                     

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
""" Building model... """

def build_model(tparams,options):
    
    #trng = RandomStreams(SEED)
    
    # Used for dropout.
    #use_noise = theano.shared(numpy_floatX(0.))

    # x: n_steps * n_x
    x = tensor.matrix('x', dtype=config.floatX)      
    n_steps = x.shape[0]                                                                              
                                             
    h_decoder = decoder_layer(tparams, x)
    
    pred = tensor.nnet.sigmoid(tensor.dot(h_decoder,tparams['Vhid']) + tparams['bhid'])
    
    f_pred = theano.function([x],pred)
    
    cost = tensor.sum(tensor.nnet.binary_crossentropy(pred,x))/n_steps                         

    return x, f_pred, cost
