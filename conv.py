import numpy as np
import onnx
import onnxruntime
import torch
import sys
import os
import pathlib 
from onnx_pytorch import code_gen
import importlib

onnx_source_folder = '/workspace/develop/model_source/big_model/model_share_0923_opset11/'
o2p_dst_folder = '/workspace/develop/o2p_models/'
subfolders = [ f.path for f in os.scandir(onnx_source_folder) if f.is_dir() ] 

p = pathlib.Path(onnx_source_folder)
# All subdirectories in the current directory, not recursive.
# subfolders2 =  [f.name for f in p.iterdir() if f.is_dir()]
subfolders2 = ['model_2056_4d0aa1_nocut']

# subfolders2.remove("model_243_59bb8f_nocut")
sys.path.append(o2p_dst_folder)
# dstfolders = [os.path.join(o2p_dst_folder, f) for f in subfolders2]

for p in subfolders2:
    srcf = os.path.join(onnx_source_folder, p, 'input', p+'.origin.onnx')
    dstf = os.path.join(o2p_dst_folder, p)
    pathlib.Path(dstf).mkdir(parents=True, exist_ok=True) 
    # print(srcf, " => ", dstf)
    # code_gen.gen(srcf, dstf, simplify_names=True)
    
    #onnx inference
    onnx_model = onnx.load(srcf)
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                       sess_options)
    
    input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input]
    if len(input_shapes) > 1:
        print("warning : more than 1 input")
    inps = [np.random.randn(*i).astype(np.float32) for i in input_shapes]

    inputs = {}
    for i in range(len(inps)):
        inputs[session.get_inputs()[i].name] =  inps[i]
        
    ort_outputs = session.run(None, inputs)
    
    # pytorch inference
    mod = importlib.import_module(p+".model")
    # from model import Model            
    pytorch_model = mod.Model()
    pytorch_model.eval()
    with torch.no_grad():
        tor_inps = [torch.from_numpy(i) for i in inps]
        torch_outputs = pytorch_model(*tor_inps)

    print(
        "Comparison result:", p, 
        np.allclose(torch_outputs[0].detach().numpy(),
                ort_outputs[0],
                atol=1e-5,
                rtol=1e-5))

    # sys.path.remove(dstf)