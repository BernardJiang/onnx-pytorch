import numpy as np
import onnx
import onnxruntime
import torch
import sys
import os
import pathlib 
from onnx_pytorch import code_gen
import importlib
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes, TrainingMode

onnx_source_folder = '/workspace/develop/model_source/big_model/model_share_0923_opset11/'
o2p_dst_folder = '/workspace/develop/o2p_models/'

sys.path.append(o2p_dst_folder)


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
    # code_gen.gen(srcf, dstf) #, simplify_names=True)
    
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
    pytorch_model = mod.Model()
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
        
        #save pytorch to onnx 
        onnxfile = dstf + "/" + p + ".onnx"
        save2onnx(pytorch_model, tor_inps, onnxfile)
        
print("total cases: ", len(subfolders2), " failed cases: ", len(failed_cases), ". ratio: ", len(failed_cases) / len(subfolders2))    
print("Failed cases: ",  failed_cases)    