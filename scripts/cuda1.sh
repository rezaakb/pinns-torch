#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python examples/ac/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > ac.txt &

wait $!

CUDA_VISIBLE_DEVICES=1 nohup python examples/kdv/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > kdv.txt &

wait $!

CUDA_VISIBLE_DEVICES=1 nohup python examples/burgers_discrete_forward/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > b_d_f.txt &

wait $!