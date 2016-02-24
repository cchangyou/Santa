#!/usr/bin/env sh

# ./build/tools/caffe train --solver=examples/mnist/lenet_solver_psgld.prototxt
# ./build/tools/caffe train --solver=examples/mnist/lenet_solver_rmsprop.prototxt
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt #--snapshot=./examples/mnist/lenet_iter_10000.solverstate
