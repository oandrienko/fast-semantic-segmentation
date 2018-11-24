r"""Utils for working with variables and proto defs."""
import re
import collections
from tensorflow.python.ops.variables import Variable
from tensorflow.python import pywrap_tensorflow
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework import ops


GraphTraversalState = collections.namedtuple(
    "GraphTraversalState", ["already_visited", "output_node_stack"])


def remove_ref_from_node_name(node_name):
    if node_name.endswith("/read"):
        node_name = node_name[:-5]
    return node_name


def node_name_matches(node_name, search_str):
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*)%s:\d+$"%search_str, node_name)
    if m:
        return m.group(1)
    return None


def node_name_from_input(node_name):
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*)?:\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def clear_node_devices(input_graph_def_nodes):
    for node in input_graph_def_nodes:
        node.device = ""


def create_var_const_map(session, var_names):
    values_dict = {}
    for key in var_names:
        tensor_name = key + ":0"
        values_dict[key] = session.run(tensor_name)
    return values_dict


def create_nodes_map(graph):
    nodes_map = {}
    for node in graph.node:
        if node.name not in nodes_map.keys():
            nodes_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected.")
    return nodes_map


def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node


def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
        new_node.input.extend([input_name])
    return new_node


def copy_variable_ref_to_graph(input_graph, output_graph,
                               var_ref, init_value, scope=''):
    if scope != '':
        new_name = (
            scope + '/' + var_ref.name[:var_ref.name.index(':')])
    else:
        new_name = var_ref.name[:var_ref.name.index(':')]
    collections = []
    for name, collection in input_graph._collections.items():
        if var_ref in collection:
            if (name == ops.GraphKeys.GLOBAL_VARIABLES or
                name == ops.GraphKeys.TRAINABLE_VARIABLES or
                scope == ''):
                collections.append(name)
            else:
                collections.append(scope + '/' + name)
    trainable = (var_ref in input_graph.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES))
    with output_graph.as_default():
        new_var = Variable(
            init_value,
            trainable,
            name=new_name,
            collections=collections,
            validate_shape=False)
        new_var.set_shape(init_value.shape)
    return new_var


def add_variable_to_graph(output_graph, var_name, init_value,
                          trainable=True, collections=[], scope=''):
    if scope != '':
        new_name = scope + '/' + var_name
    else:
        new_name = var_name

    with output_graph.as_default():
        new_var = Variable(
            init_value,
            trainable,
            name=new_name,
            collections=collections,
            validate_shape=False)
        new_var.set_shape(init_value.shape)
    return new_var


def get_vars_from_checkpoint(session, checkpoint, checkpoint_version):
    var_list = {}
    meta_graph_file = checkpoint + '.meta'
    saver = saver_lib.import_meta_graph(meta_graph_file)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        try:
            tensor_name = key + ":0"
            tensor = session.graph.get_tensor_by_name(
                tensor_name)
        except KeyError:
            continue
        var_list[key] = tensor
    saver.restore(session, checkpoint)
    return var_list
