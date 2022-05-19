import numpy as np
import onnx
import onnxruntime
import torch
import torch.fx as fx
import torch.nn as nn
import sys
import os
import pathlib 
from onnx_pytorch import code_gen
import importlib
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes, TrainingMode
from kqconv import KQConv2d
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import operator

onnx_source_folder = '/workspace/develop/dataset/model_source/big_model/model_share_0923_opset11/'
o2p_dst_folder = '/workspace/develop/dataset/o2p_models/'

sys.path.append(o2p_dst_folder)


def matchType(patterntype, node: fx.Node, modules: Dict[str, Any]):
    if not isinstance(node, fx.Node):
        return False
    if node.op != 'call_module':
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False
    if type(modules[node.target]) is not patterntype:
        return False
    return True


def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def old_pattern(n_conv_1_zeropad2d, getattr_1):
    return torch.conv2d(n_conv_1_zeropad2d, getattr_1, None, (2, 2))

# Define the replacement (same rules as the pattern)
def replacement(n_conv_1_zeropad2d, getattr_1):
    input_data = torch.clamp(n_conv_1_zeropad2d, min=-1.1e+10, max=1e+10)
    weight = torch.clamp(getattr_1, min=-1e+10, max=1e+10)    
    return torch.conv2d(input_data, weight, None, (2, 2))


                
def transform(m: torch.nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)
    traced = fx.symbolic_trace(m)
    modules = dict(traced.named_modules())
    
    # Step 2: Modify this Graph or create a new one
    # patterns = set([nn.Conv2d, nn.Linear])
    patterns = set([operator.add, torch.add, "add"])
    
    # Replace `old_pattern` with `replacement` in `traced`
    # fx.replace_pattern(traced, old_pattern, replacement)

    # Go through all the nodes in the Graph
    for n in traced.graph.nodes:
        # If the target matches one of the patterns
        if any(n.target == pattern for pattern in patterns):
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with traced.graph.inserting_after(n): 
                new_node = traced.graph.call_function(torch.sub, n.args, {**n.kwargs, 'alpha': -1.}) #n.kwargs
                n.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            traced.graph.erase_node(n)           

    traced.graph.lint()    
    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, traced.graph)

def compare2onnx(onnx_model, onnx_model2):
    for (f, f2) in zip(onnx_model.graph.input, onnx_model2.graph.input):
        if f.type != f2.type: 
            print("Input Diff: ", f, f2)
                 
    for (f, f2) in zip(onnx_model.graph.output, onnx_model2.graph.output):
        if f.type != f2.type: 
            print("Output Diff: ", f, f2)
                 
    for (f, f2) in zip(onnx_model.graph.initializer, onnx_model2.graph.initializer):
        if f.type != f2.type: 
            print("initializer Diff: ", f, f2)                        
            
    for (f, f2) in zip(onnx_model.graph.node, onnx_model2.graph.node):
        if f.type != f2.type: 
            print("node Diff: ", f, f2)                        
            
 

def save2onnx(model_orig, img, onnx_export_file, disable_quantization=False):
    
    try:
        
        # if disable_quantization:
            # disable quantization before saving to onnx.
            # for m in model_orig.modules():
                # if isinstance(m, QConv2d) or isinstance(m, QLinear):
                    # m.quantize = False
            # qparams = get_quantized_model_and_params(model_orig)
            # filename_json = onnx_export_file + ".json"
            # with open(filename_json, "w") as fp:
                # json.dump(qparams, fp, indent=4)

        # onnx_export_file = result_folder+'mobilenetv2_zeroq.onnx'
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        print('****onnx file****',onnx_export_file)
        model_orig.eval()
        # model = copy.deepcopy(model_orig)
        model = model_orig
        # for name, param in model.named_parameters():
        #     print("name = ", name)
        #     param.requires_grad = False 
        # img = torch.zeros((1, 3, 224, 224))
        model.eval()
        y = model(*img)  # dry run
        # torch.onnx.export(  output_names=['classes', 'boxes'] if y is None else ['output'])
        torch.onnx.export(model,               # model being run
                            *img,                         # model input (or a tuple for multiple inputs)
                            onnx_export_file,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=11,          # the ONNX version to export the model to
                            do_constant_folding=False,  # whether to execute constant folding for optimization
                            input_names = ['images'],   # the model's input names
                            output_names = ['classes', 'boxes'] if y is None else ['output'], # the model's output names
                            training=TrainingMode.PRESERVE,
                            keep_initializers_as_inputs=True,
                            verbose=False
        )     # Checks
        onnx_model = onnx.load(onnx_export_file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % onnx_export_file)
    except Exception as e:
        print('ONNX export failure: %s' % e)


subfolders = [ f.path for f in os.scandir(onnx_source_folder) if f.is_dir() ] 

p = pathlib.Path(onnx_source_folder)
# All subdirectories in the current directory, not recursive.
# subfolders2 =  [f.name for f in p.iterdir() if f.is_dir()]
# subfolders2 = ['model_216_294e55_cut_sigmoid', 'model_215_135555_cut_sigmoid', 'model_094_9e8715_cut_sigmoid', 'model_211_c2b850_nocut', 'model_212_0c51ee_nocut', 'model_220_d31163_nocut', 'model_211_c2b850_cut_sigmoid', 'model_223_da3b11_cut_sigmoid', 'model_215_135555_nocut', 'model_228_099330_cut_sigmoid', 'model_073_70211d_nocut', 'model_093_68cdc5_nocut', 'model_073_70211d_cut_sigmoid', 'model_021_0105ae_nocut', 'model_234_9e7e66_cut_sigmoid', 'model_252_099330_cut_sigmoid', 'model_212_0c51ee_cut_sigmoid', 'model_220_d31163_cut_sigmoid', 'model_223_da3b11_nocut', 'model_094_9e8715_nocut', 'model_235_0873f9_nocut', 'model_216_294e55_nocut', 'model_225_4c8c3f_cut_sigmoid', 'model_252_099330_nocut', 'model_093_68cdc5_cut_sigmoid', 'model_235_0873f9_cut_sigmoid', 'model_228_099330_nocut', 'model_225_4c8c3f_nocut', 'model_234_9e7e66_nocut']
subfolders2 = ['model_073_70211d_nocut']

failed_cases = []
for p in subfolders2:
    srcf = os.path.join(onnx_source_folder, p, 'input', p+'.origin.onnx')
    dstf = os.path.join(o2p_dst_folder, p)
    pathlib.Path(dstf).mkdir(parents=True, exist_ok=True) 
    # print(srcf, " => ", dstf)
    code_gen.gen(srcf, dstf, simplify_names=True)
    
    #onnx inference
    onnx_model = onnx.load(srcf)
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                       sess_options)
    
    input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input]


    inps = [np.random.randn(*i).astype(np.float32) for i in input_shapes]

    inputs = { session.get_inputs()[i].name : inps[i] for i in range(len(inps)) }

        
    ort_outputs = session.run(None, inputs)
    
    # pytorch inference
    mod = importlib.import_module(p+".model")
    # from model import Model            
    pytorch_model_1 = mod.Model()
    pytorch_model_2 = transform(pytorch_model_1)
    pytorch_model_2.to_folder(dstf, "transformed_model")
    mod_2 = importlib.import_module(p+".module")
    pytorch_model = mod_2.transformed_model()
    # pytorch_model = pytorch_model_2
    
    pytorch_model.eval()
    with torch.no_grad():
        tor_inps = [torch.from_numpy(i) for i in inps]
        torch_outputs = pytorch_model(*tor_inps)
        
    test_result = np.allclose(torch_outputs[0].detach().numpy(),
        ort_outputs[0],
        atol=1e-3,
        rtol=1e-3)        

    absdiff = abs(torch_outputs[0].detach().numpy()-ort_outputs[0]).max()
    print("Comparison result:", p, test_result, absdiff)
 
    if not test_result:
        failed_cases.append(p)
        
        ptfile = dstf + "/" + p + ".pt"
        torch.save(pytorch_model.state_dict(),ptfile)
        
        #save pytorch to onnx 
        onnxfile = dstf + "/" + p + ".onnx"
        # save2onnx(pytorch_model, tor_inps, onnxfile)
        
        onnx_model2 = onnx.load(onnxfile)
        sess_options2 = onnxruntime.SessionOptions()
        session2 = onnxruntime.InferenceSession(onnx_model2.SerializeToString(),
                                       sess_options2)
        ort_outputs2 = session.run(None, inputs)
        test_result2 = np.allclose(ort_outputs[0],
            ort_outputs2[0],
            atol=1e-3,
            rtol=1e-3)        

        absdiff2 = abs(torch_outputs[0].detach().numpy()-ort_outputs[0]).max()
        print("Comparison result onnx vs onnx:", p, test_result2, absdiff2)
        # compare2onnx(onnx_model, onnx_model2)
    
    
        
print("total cases: ", len(subfolders2), " failed cases: ", len(failed_cases), ". ratio: ", len(failed_cases) / len(subfolders2))    
print("Failed cases: ",  failed_cases)    