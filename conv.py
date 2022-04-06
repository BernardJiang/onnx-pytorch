import numpy as np
import onnx
import onnxruntime
import torch
import sys
import os
import pathlib 
from onnx_pytorch import code_gen

onnx_source_folder = '/workspace/develop/model_source/big_model/model_share_0923_opset11/'
o2p_dst_folder = '/workspace/develop/o2p_models/'
subfolders = [ f.path for f in os.scandir(onnx_source_folder) if f.is_dir() ] 

p = pathlib.Path(onnx_source_folder)
# All subdirectories in the current directory, not recursive.
subfolders2 =  [f.name for f in p.iterdir() if f.is_dir()]

# subfolders2.remove("model_243_59bb8f_nocut")

# dstfolders = [os.path.join(o2p_dst_folder, f) for f in subfolders2]
for p in subfolders2:
    srcf = os.path.join(onnx_source_folder, p, 'input', p+'.origin.onnx')
    dstf = os.path.join(o2p_dst_folder, p)
    pathlib.Path(dstf).mkdir(parents=True, exist_ok=True) 
    print(srcf, " => ", dstf)
    code_gen.gen(srcf, dstf)
    
