import os
import sys
import ctypes
import numpy as np
from datetime import datetime as dt
from cuda import cudart
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.compat.v1.disable_eager_execution()

np.random.seed(97)
# tf.compat.v1.set_random_seed(97)
epsilon = 1e-6
onnxFile = "./encoderV2.onnx"
onnxSurgeonFile = "./encoderV2-surgeon.onnx"
soFile = "../LayerNormPlugin.so"
trtFile = "./encoder.plan"

np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

def check(a, b, weak=False, info=""):  # 用于比较 TF 和 TRT 的输出结果
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:" % info, res, diff0, diff1)

def printArray(x, info="", n=5):  # 用于输出数组统计信息
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])
    #print('\t',x.reshape(-1)[:n])

# 将 .onnx 文件中 TensorRT 不原生支持的节点替换为 Plugin ----------------------------

# graph = gs.import_onnx(onnx.load(onnxFile))

# for node in graph.nodes:
#     if node.name == 'Add_104':
#         pluginV = gs.Variable("MyL2normPluginVariable-0", np.dtype(np.float32), None)
#         pluginN = gs.Node("l2norm", "MyL2normPluginNode-0", inputs=[node.outputs[0]], outputs=[pluginV], attrs={})
#         graph.nodes.append(pluginN)

#     if node.name == 'Add_115':
#         node.inputs.clear()
#         node.inputs = pluginN.outputs

# graph.cleanup().toposort()
# onnx.save(gs.export_onnx(graph), onnxSurgeonFile)
# print("Succeeded inserting L2norm node!")

# TensorRT 中加载 .onnx 和 .so 创建 engine ---------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

config.flags = 1 << int(trt.BuilderFlag.FP16)

config.max_workspace_size = 8 << 30
parser = trt.OnnxParser(network, logger)
with open(onnxFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing onnx file!")

inputTensor = [network.get_input(i) for i in range(network.num_inputs)]
profile.set_shape(inputTensor[0].name, [1, 16, 80], [16, 64, 80], [64, 256, 80])
profile.set_shape(inputTensor[1].name, [1,], [16,], [64,])

config.add_optimization_profile(profile)
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write(engineString)

