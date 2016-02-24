# Santa Algorithm for RNN training

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* [Theano 0.7](http://deeplearning.net/software/theano/)
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)

This code runs on CPU since the dataset is relatively small, but it is easy to make the code run on GPU if desired.

## How to use the code

Running Santa Algorithm:

    python eval_music_santa.py sys.argv[1] sys.argv[2]

Two examples:
    python eval_music_santa.py Piano large_lrate
    python eval_music_santa.py JSB small_lrate

sys.argv[1] indicates which dataset to use, sys.argv[2] indicates which learning rate to use, we only specify two learning rates used in the paper, i.e. 1.*1e-3 and 2.*1e-4.


Running other optimization Algorithms (i.e. SGD, SGD-M, RMSprop & Adam):

    python eval_music.py sys.argv[1] sys.argv[2]

Two examples:
    python eval_music.py Piano SGD
    python eval_music.py JSB Adam

sys.argv[1] indicates which dataset to use, sys.argv[2] indicates which algorithm to use.

## Output

In order to monitor the training process, all the output information will be printed out in the .log file.
The final parameters we learned will be stored in a .npz file. 


