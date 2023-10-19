#!/bin/bash

CUDA_VISIBLE_DEVICES=3 nohup python examples/navier_stokes/train.py trainer.devices=[0] callbacks=null logger=null model.amp=true model.cudagraph_compile=true model.jit_compile=true trainer.enable_progress_bar=false > navier3.txt &

wait $!

CUDA_VISIBLE_DEVICES=3 nohup python examples/schrodinger/train.py trainer.devices=[0] callbacks=null logger=null model.amp=true model.cudagraph_compile=true model.jit_compile=true trainer.enable_progress_bar=false > sch.txt &

wait $!

CUDA_VISIBLE_DEVICES=3 nohup python examples/ac/train.py trainer.devices=[0] callbacks=null logger=null model.amp=false model.cudagraph_compile=true model.jit_compile=true trainer.enable_progress_bar=false > ac.txt &

wait $!

CUDA_VISIBLE_DEVICES=3 nohup python examples/kdv/train.py trainer.devices=[0] callbacks=null logger=null model.amp=true model.cudagraph_compile=true model.jit_compile=true trainer.enable_progress_bar=false > kdv.txt &

wait $!