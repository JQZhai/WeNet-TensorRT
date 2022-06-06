from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs
def example_addNode():
    tensor0 = gs.Variable(name="tensor0", dtype=np.float32, shape=['B', 3, 64, 64])
    tensor1 = gs.Variable(name="tensor1", dtype=np.float32, shape=None)
    tensor2 = gs.Variable(name="tensor2", dtype=np.float32, shape=None)

    node0 = gs.Node(name="myIdentity0", op="Identity", inputs=[tensor0], outputs=[tensor1])
    node1 = gs.Node(name="myIdentity1", op="Identity", inputs=[tensor1], outputs=[tensor2])

    graph = gs.Graph(nodes=[node0, node1], inputs=[tensor0], outputs=[tensor2])
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "model-02-01.onnx")

    for node in graph.nodes:
        if node.op == 'Identity' and node.name == 'myIdentity0':  # 遍历计算图找到需要添加节点的位置
            constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))  # 构造新节点和新张量
            tensor3 = gs.Variable(name="tensor3", dtype=np.float32, shape=None)
            newNode = gs.Node(name="myAdd", op="Add", inputs=[node.outputs[0], constant0], outputs=[tensor3])

            graph.nodes.append(newNode)  # 记得把新节点加入计算图中
            index = node.o().inputs.index(node.outputs[0])  # 小心地找到下一个节点中对应输入张量的位置
            node.o().inputs[index] = tensor3  # 替换为新张量


    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "model-02-02.onnx")



def wenet_encoder_slice79_cast(graph):
    t1 = [node for node in graph.nodes if node.name == "Slice_84"][0]
    t2 = [node for node in graph.nodes if node.name == "Not_30"][0]
    t3 = [node for node in graph.nodes if node.name == "Slice_79"][0]
    target_constant = ["Constant_75","Constant_76","Constant_77","Constant_78"]
    constants = [node for node in graph.nodes if node.name in target_constant]

    cast_out = gs.Variable("cast_out", dtype=np.int64)
    cast = gs.Node(op="Cast", name="cast?", attrs={'to' : 7, }, inputs=t2.outputs, outputs=[cast_out])
    graph.nodes.append(cast)

    t3.inputs.clear()
    t3.inputs = [cast.outputs[0], constants[1].outputs[0],constants[2].outputs[0], constants[0].outputs[0], constants[3].outputs[0]]


    target_wheres = ["Where_196", "Where_208", "Where_353", "Where_365", "Where_510", "Where_522", "Where_667",
    "Where_679", "Where_824", "Where_836", "Where_981", "Where_993", "Where_1138", "Where_1150", "Where_1295", "Where_1307", 
    "Where_1452", "Where_1464", "Where_1609", "Where_1621", "Where_1766", "Where_1778", "Where_1923", "Where_1935"]
    wheres = [node for node in graph.nodes if node.name in target_wheres]

    target_nots = ["Not_193", "Not_204", "Not_350", "Not_361", "Not_507", "Not_518", "Not_664", "Not_657", "Not_821", 
    "Not_832", "Not_978", "Not_989", "Not_1135", "Not_1146", "Not_1292", "Not_1303", "Not_1449", "Not_1460", "Not_1606", "Not_1617",
    "Not_1763", "Not_1774", "Not_1920", "Not_1931"]
    nots = [node for node in graph.nodes if node.name in target_nots]

    target_casts = ["Cast_194", "Cast_206", "Cast_351", "Cast_363", "Cast_508", "Cast_520", "Cast_665", "Cast_677", "Cast_822", "Cast_834", 
    "Cast_979", "Cast_991", "Cast_1136", "Cast_1148", "Cast_1293", "Cast_1305", "Cast_1450", "Cast_1462", "Cast_1607", "Cast_1619", 
    "Cast_1764", "Cast_1776", "Cast_1921", "Cast_1933"]
    casts = [node for node in graph.nodes if node.name in target_casts]

    for i in range(len(nots)):
        not_in = nots[i].inputs
        not_out = nots[i].outputs
        cast_in = casts[i].inputs
        cast_out = casts[i].outputs

        casts[i].inputs = not_in
        nots[i].inputs = cast_out
        out1 = wheres[i].i(1).outputs[0]
        out2 = wheres[i].i(2).outputs[0]
        wheres[i].inputs.clear()
        wheres[i].inputs = [not_out[0], out1, out2]

    graph.cleanup().toposort()
    return graph


def start_unqueeze_change(graph):
    node_list = {}
    for node in graph.nodes:
        if node.op == 'Unsqueeze' and node.name == "Unsqueeze_34":
            node_list["Unsqueeze_34"] = node

        if node.op == 'Sub' and node.name == "Sub_31":
            node_list["Sub_31"] = node

        if node.op == 'Mul' and node.name == "Mul_32":
            node_list["Mul_32"] = node

    end_node   = node_list["Unsqueeze_34"].outputs[0]
    start_node = node_list["Sub_31"].inputs[0]

    temp_node = node_list["Unsqueeze_34"].inputs[0]
    node_list["Unsqueeze_34"].inputs[0] = start_node
    node_list["Mul_32"].outputs[0] =  end_node

    node_list["Unsqueeze_34"].outputs[0] = temp_node
    node_list["Sub_31"].inputs[0] = temp_node
    return graph

def find_node(graph,type,name):
    for node in graph.nodes:
        if node.op == type and node.name == name:
           return node


def remove_node(graph,type,name):
    node = find_node(graph, type, name)
    node.inputs = []
    node.outputs = []
    graph.nodes.remove(node)
    return graph

def reshape_conv(graph):


    ## add transpose and suqeeze
    Mul_64 = find_node(graph, "Mul", "Mul_64")
    Mul_64_ori_end_node = Mul_64.outputs[0]
    Mul_64.outputs = []
    Mul_64.inputs = []

    start_node = gs.Variable(name="Relu_38_output", dtype=None, shape=None)
    Relu_38 = find_node(graph, "Relu", "Relu_38")
    Relu_38.outputs[0] = start_node

    MatMul_61 = find_node(graph, "MatMul", "MatMul_61")
    conv_weight = np.transpose(MatMul_61.inputs[1].values, (1, 0)).reshape((-1, 256, 1, 19)) *16
    conv_weight = gs.Constant(name="ReshapeConv1_weight,", values=conv_weight )

    Add_62 = find_node(graph, "Add", "Add_62")
    conv_bias = Add_62.inputs[0].values *16
    conv_bias = gs.Constant(name="ReshapeConv1_bias,", values=conv_bias )

    Conv_37 = find_node(graph, "Conv", "Conv_37")
    newConv = Conv_37.copy()

    newConv.attrs['kernel_shape'] = [1, 19]
    newConv.attrs['strides'] = [1, 1]
    newConv.name = "ReshapeConv1"

    newConv.inputs = [start_node,conv_weight,conv_bias]

    end_node = Add_62.outputs[0]
    Add_62.outputs = []
    newConv.outputs = [end_node]

    #
    # #
    # #
    Mul_64_output = gs.Variable(name="Mul_64_output", dtype=None, shape=None)
    Mul_64.outputs = [Mul_64_output]
    #
    ReshapeConv1Transpose         = find_node(graph, "Transpose", "Transpose_51").copy()
    ReshapeConv1Transpose.name    = "ReshapeConv1Transpose"
    ReshapeConv1Transpose_param   = gs.Constant(name="ReshapeConv1Transpose_param,", values=np.array([0,2,1,3],dtype=np.int32))
    ReshapeConv1Transpose.inputs  = [end_node,ReshapeConv1Transpose_param]
    ReshapeConv1Transpose_output  = gs.Variable(name="ReshapeConv1Transpose_output", dtype=None, shape=None)
    ReshapeConv1Transpose.outputs = [ReshapeConv1Transpose_output]

    ReshapeConv1Unsqueeze_param = gs.Constant(name="ReshapeConv1Unsqueeze_param,",
                                              values=np.array([-1], dtype=np.int32))
    ReshapeConv1Unsqueeze = gs.Node(name="ReshapeConv1Unsqueeze", op="Squeeze", inputs=[ReshapeConv1Transpose_output,ReshapeConv1Unsqueeze_param],
            outputs=[Mul_64_ori_end_node])
    nodes_temp = [newConv,ReshapeConv1Transpose,ReshapeConv1Unsqueeze]

    graph.nodes+=nodes_temp
    graph.cleanup().toposort()
    return graph


def layer_norm_change(graph):

    ReduceMean_85 = find_node(graph, "ReduceMean", "ReduceMean_85")
    ReduceMean_85.attrs['axes'] = 1

    graph.cleanup().toposort()
    return graph
    pass

def change_Shape_65(graph):
    ReshapeConv1 = find_node(graph, "Conv", "ReshapeConv1")
    Shape_65 = find_node(graph, "Shape", "Shape_65")
    Shape_65.inputs = ReshapeConv1.outputs

    Gather_67 = find_node(graph, "Gather", "Gather_67")
    Gather_67.inputs[1] = gs.Constant(name="Gather_67_param,", values=np.array([2],dtype=np.int32))
    graph.cleanup().toposort()
    return graph
    pass




def Slice_74_change(graph):
    ### only slice to Shape_609
    Slice_74   = find_node(graph, "Slice", "Slice_74")
    Slice_74_outputs = gs.Variable(name="Slice_74_outputs", dtype=None, shape=None)
    Slice_74.outputs[0] = Slice_74_outputs
    Shape_609  = find_node(graph, "Shape", "Shape_609")
    # Shape_609.inputs[0] = Slice_74_outputs
    Shape_609.inputs = []



    def chang_reshape_inputs(graph,name):
        node = find_node(graph, "Reshape", name)
        MatMul_node_outpts = node.o().outputs[0]
        node.o().outputs = []
        MatMul = node.inputs[0]
        MatMul_node = MatMul.inputs[0]
        MatMul_weight = MatMul_node.inputs[1].values
        # MatMul_node_outpts = MatMul_node.outputs[0]
        MatMul_node.outputs = []
        MatMul_node.inputs = []
        return MatMul_weight,MatMul_node_outpts

    name_list = ["Reshape_145","Reshape_302","Reshape_459","Reshape_616","Reshape_773","Reshape_930",
                 "Reshape_1087","Reshape_1244", "Reshape_1401", "Reshape_1558","Reshape_1715","Reshape_1872"]
    Concat_615 = find_node(graph, "Concat", "Concat_615")
    Concat_615_param0  = gs.Constant(name="Concat_615_param0", values=np.array([len(name_list)*64],dtype=np.int64))
    Concat_615.inputs[-1] = Concat_615_param0

    ##MatMul_ALL
    MatMul_weight_node_outpts_list = [chang_reshape_inputs(graph,name) for name in name_list]

    MatMul_weight_list =[value[0] for value in MatMul_weight_node_outpts_list]
    Slice_74_weight_value = np.load("tensor.npy")
    """
   这里错了
    MatMul_weight = np.concatenate(MatMul_weight_list,1)
    Slice_74_weight_value = np.dot(Slice_74_weight_value,MatMul_weight)
    Slice_74_weight_value = np.reshape(Slice_74_weight_value,(1,5000,4,768)).transpose((0,2,3,1) )
    """
    ###这是修改的
    Slice_74_weight_value_list = []
    for MatMul_weight in MatMul_weight_list:
        Slice_74_weight_value_list.append(
            np.dot(Slice_74_weight_value, MatMul_weight).reshape((1, 5000, 4, 64)).transpose((0, 2, 3, 1))
        )
    Slice_74_weight_value = np.concatenate(Slice_74_weight_value_list, 2)

    Slice_74_weight = gs.Constant(name="Slice_74_weight", values=Slice_74_weight_value)
    Slice_74.inputs[0] = Slice_74_weight
    Slice_74.inputs[3] = gs.Constant(name="Slice_74_axis", values=np.array([3],dtype = np.int32))

    Reshape_616 = find_node(graph, "Reshape", "Reshape_616")
    Reshape_616.inputs[0] = Slice_74_outputs
    #
    #MatMul
    # MatMul_SliceALL_weight = gs.Constant(name="MatMul_SliceALL_weight", values=MatMul_weight)
    # MatMul_SliceALL = find_node(graph, "MatMul", "MatMul_141").copy()
    # MatMul_SliceALL.name ='MatMul_SliceALL'
    # MatMul_SliceALL.inputs = [Slice_74_outputs,MatMul_SliceALL_weight ]
    # MatMul_SliceALL.outputs = [Reshape_616_inputs]
    # graph.nodes.append(MatMul_SliceALL)

    ###slice
    Transpose_623 = find_node(graph, "Transpose", "Transpose_623")
    # MatMul_Slice_inputs = gs.Variable(name="Transpose_623_outputs", dtype=None, shape=None)
    # Transpose_623.outputs = [MatMul_Slice_inputs]
    Transpose_623.outputs = []
    MatMul_outputs_list = [value[1] for value in MatMul_weight_node_outpts_list]
    for i,MatMul_outputs in enumerate(MatMul_outputs_list):
        Slice_node = find_node(graph, "Slice", "Slice_74").copy()
        Slice_node_param = [
            gs.Constant(name="MatMul_Slice_{}_starts".format(i), values=np.array([i*64],dtype = np.int32)),
            gs.Constant(name="MatMul_Slice_{}_ends".format(i), values=np.array([(i+1)*64],dtype = np.int32) ),
            gs.Constant(name="MatMul_Slice_{}_axes".format(i), values=np.array([2],dtype = np.int32)  ),
            gs.Constant(name="MatMul_Slice_{}_steps".format(i), values=np.array([1],dtype = np.int32) ),
        ]
        Slice_node.inputs = [Slice_74_outputs]+Slice_node_param
        Slice_node.outputs = [MatMul_outputs]
        Slice_node.name = "MatMul_Slice_{}".format(i)

        # MatMul_Gather_index = gs.Constant(name="MatMul_Gather_{}_index".format(i), values=np.array([i], dtype=np.int32))
        # Gather = find_node(graph, "Gather", "Gather_611").copy()
        # Gather.inputs = [MatMul_Slice_inputs,MatMul_Gather_index]
        # Gather.outputs = [MatMul_outputs]
        # Gather.attrs['axis'] = 0
        # Gather.name = "MatMul_Gather_{}".format(i)

        graph.nodes.append(Slice_node)
    return graph

if __name__ == '__main__':

    encoder = gs.import_onnx(onnx.load("/workspace/encoder.onnx") )
    encoder = wenet_encoder_slice79_cast(encoder)
    # encoder = start_unqueeze_change(encoder)
    encoder = reshape_conv(encoder)
    # encoder = change_Shape_65(encoder)
    encoder = Slice_74_change(encoder)
    encoder.cleanup().toposort()
    onnx.save( gs.export_onnx(encoder), "encoder3.onnx")