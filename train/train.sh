mkdir reconnet_0_01 #Create a folder to save the caffe models (measurement rate is 0.01)

../../../build/tools/caffe train --solver ReconNet_solver.prototxt
