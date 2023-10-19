#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python examples/schrodinger/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=false,true model.jit_compile=true trainer.enable_progress_bar=false --multirun > navier2.txt &

wait $!
