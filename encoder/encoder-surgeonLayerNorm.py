#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnxFilePath = "/workspace/zjq/"
sourceOnnx = "./encoder3.onnx"
destinationOnnx = "./encoderV2.onnx"

bLayerNormPlugin = True
nLayerNormPlugin = 0
epsilon = 1e-5

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

if bLayerNormPlugin:
    for node in graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

            inputTensor = node.inputs[0]
            lastDivNode = node.o().o(0).o().o().o().o()

            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=[inputTensor],
            outputs=[lastDivNode.outputs[0]], attrs={"epsilon": float(epsilon)})
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1

            lastDivNode.outputs = []
            continue

nCast = 0
nGather = 0
target1 = ['Cast_162', 'Cast_319', 'Cast_476', 'Cast_633', 'Cast_790', 'Cast_947', 'Cast_1104', 'Cast_1261',
            'Cast_1418', 'Cast_1575', 'Cast_1732', 'Cast_1889', 'Cast_1933', 'Cast_1983', 'Cast_1983' ]
target2 = ['Gather_118', 'Gather_275', 'Gather_432', 'Gather_589', 'Gather_746', 'Gather_903', 'Gather_1060',
            'Gather_1217', 'Gather_1374', 'Gather_1531', 'Gather_1688', 'Gather_1845']
nConstant = 0
for node in graph.nodes:
    if node.op == 'Equal' and node.o(0).op == 'Cast' and node.o(1).op == 'Cast':
        nCast += 1
        In0 = node.outputs[0]
        
        In01 = node.o(0).o().inputs[1]
        In02 = node.o(0).o().inputs[2]
        node.o(0).o().inputs = [In0, In01, In02]
        
        In11 = node.o(1).o().inputs[1]
        In12 = node.o(1).o().inputs[2]
        node.o(1).o().inputs = [In0, In11, In12]
    
    if node.name == 'Cast_6':
            nCast += 1
            node.o().inputs[1] = node.inputs[0]
    if node.name in target1:
        nCast +=1
        node.o().inputs[0] = node.i().outputs[0]

    if node.name in target2:
        nGather +=1
        node.o(1).o().o().inputs[1] = node.o(0).o().outputs[0]
        node.o(2).o().o().inputs[1] = node.o(0).o().outputs[0]

    if 'Where' in node.name and node.o().op == 'Softmax':
        t1 = gs.Constant(str(nConstant)+'_fp16_const', np.ascontiguousarray(np.array([-1e4], dtype=np.float32)))
        nConstant += 1
        in1 = node.i(0).outputs[0]
        in2 = node.i(2).outputs[0]
        node.inputs.clear()
        node.inputs = [in1, t1, in2]

graph.cleanup()
onnx.save(gs.export_onnx(graph), destinationOnnx)


print("finish encoder onnx-graphsurgeon!")
print("%4d ClearCast" %nCast)
print("%4d ClearGather" %nGather)
print("%4d ClearConstant" %nConstant)
print("%4d LayerNormPlugin" %nLayerNormPlugin)
