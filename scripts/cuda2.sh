#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python examples/burgers_continuous_inverse/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > b1.txt &

wait $!

CUDA_VISIBLE_DEVICES=2 nohup python examples/burgers_discrete_inverse/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > b2.txt &

wait $!

CUDA_VISIBLE_DEVICES=2 nohup python examples/burgers_continuous_forward/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > b3.txt &

wait $!

CUDA_VISIBLE_DEVICES=3 python examples/burgers_continuous_forward/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false model.jit_compile=true