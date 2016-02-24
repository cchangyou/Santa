#!/usr/bin/env sh

./build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt 2>>/data/results_ccy/imagenet/sgdm.log #--snapshot=models/bvlc_alexnet/caffe_alexnet_train_iter_40000.solverstate
# ./build/tools/caffe train --solver=examples/mnist/lenet_solver_rmsprop.prototxt
# ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt --snapshot=./examples/mnist/lenet_iter_10000.solverstate
