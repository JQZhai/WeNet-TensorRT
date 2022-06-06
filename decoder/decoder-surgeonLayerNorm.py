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

onnxFilePath = "/workspace/"
sourceOnnx = onnxFilePath + "decoder.onnx"
destinationOnnx = "./decoderV2.onnx"

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
target1 = ['Cast_214', 'Cast_218', 'Cast_276', 'Cast_280', 'Cast_355', 'Cast_359', 'Cast_417',
            'Cast_421', 'Cast_496', 'Cast_500', 'Cast_558', 'Cast_562', 'Cast_637', 'Cast_641',
            'Cast_699', 'Cast_703', 'Cast_778', 'Cast_782', 'Cast_840', 'Cast_844', 'Cast_919',
            'Cast_923', 'Cast_981', 'Cast_985']
nGather = 0
target2 = ['Gather_181', 'Gather_243', 'Gather_322', 'Gather_384', 'Gather_463', 'Gather_525', 'Gather_604', 'Gather_666',
            'Gather_745', 'Gather_807', 'Gather_886', 'Gather_948']

nConstant = 0
for node in graph.nodes:
    if node.name == 'Cast_26' or node.name == 'Cast_94' or node.name == 'Cast_127' or node.name == 'Cast_1042':
        nCast += 1
        node.o().inputs[1] = node.inputs[0]
    if node.name == 'Cast_119':
        nCast += 1
        node.o().inputs = node.inputs
    if node.name == 'Cast_150':
        nCast += 1
        node.o().inputs[0] = node.inputs[0]
    if node.name == 'Cast_151':
        nCast += 1
        node.o().inputs[1] = node.inputs[0]
    if node.name == 'And_152':
        nCast += 1
        node.o().o(1).o().inputs = node.o().o(0).o().outputs
        node.o().o(2).o().inputs = node.o().o(0).o().outputs
        node.o().o(3).o().inputs = node.o().o(0).o().outputs
        node.o().o(4).o().inputs = node.o().o(0).o().outputs
        node.o().o(5).o().inputs = node.o().o(0).o().outputs
    if node.name in target1:
        nCast += 1
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
