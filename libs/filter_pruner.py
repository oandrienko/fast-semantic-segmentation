r""" Filter Pruner for model compression

See the paper "Pruning Filters for Efficient ConvNets" for more details
    https://arxiv.org/abs/1608.08710

Usage:

    compressor = FilterPruner(input_node=input_node_name,
        output_node=output_node_name, compression_factor=compression_factor)

    compressor.compress(input_graph_def, FLAGS.input_checkpoint)

    compressor.save(
        output_checkpoint_dir=FLAGS.output_dir,
        output_checkpoint_name=output_path_name)
"""
import os
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import graph_utils
from graph_utils import GraphTraversalState


FilterPrunerNodeSpec = collections.namedtuple(
    "FilterPrunerNodeSpec", ["source", "target", "following"])


def plot_magnitude_of_weights(plot_name, weights, compression):
    weight_magnitudes = np.sort(np.abs(weights).sum((0,1,2)))
    weight_magnitudes_list = weight_magnitudes.tolist()
    _, _, _, channel_dim  = weights.shape
    cut_off_x = channel_dim - int(channel_dim*compression)
    cut_off_y = max(weight_magnitudes_list)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(weight_magnitudes_list)
    plt.xlabel("Output channel")
    plt.xlim([0,channel_dim])
    plt.ylabel("L1 norm")
    plt.title(plot_name)
    plt.axvline(x=cut_off_x, ymin=0, ymax=cut_off_y,
        color='red', zorder=2)
    plt.xticks(list(plt.xticks()[0])+[cut_off_x])
    plt.show()


class FilterPruner(object):

    def __init__(self,
                 input_node,
                 output_node,
                 compression_factor,
                 init_pruner_specs=None,
                 skippable_nodes=[],
                 checkpoint_version=tf.train.SaverDef.V2,
                 clear_devices=True,
                 interactive_mode=False,
                 pruner_mode="ICNetPruner",
                 soft_apply=False):
        self.input_node = input_node
        self.output_node = output_node
        self.compression_factor = compression_factor
        self.checkpoint_version = checkpoint_version
        self.clear_devices = clear_devices
        self.skippable_nodes = skippable_nodes
        self.init_pruner_specs = (init_pruner_specs
                if init_pruner_specs is not None else {})
        # Empty members filled on call to prune
        self.pruner_specs = [] # Must be updated in order
        self.nodes_map = {}
        self.output_graph = None
        self.output_graph_def = None
        if pruner_mode != "ICNetPruner":
            raise ValueError("Currently only the pruner mode `ICNetPruner`"
                             " is implemented for pruning ICNet filters.")
        self.mode = pruner_mode
        self.interactive_mode = interactive_mode
        self.soft_apply = soft_apply

    def _init_pruning_graph(self, input_checkpoint):
        # Import graph def to use in the session
        session = tf.Session()
        vars_dict = graph_utils.get_vars_from_checkpoint(
                    session, input_checkpoint, self.checkpoint_version)
        self.input_graph_def = tf.get_default_graph().as_graph_def()
        # Grab all the variables found in the checkpoint
        self.trainable_vars = vars_dict.keys()
        self.nodes_map = graph_utils.create_nodes_map(self.input_graph_def)
        self.values_map = graph_utils.create_var_const_map(
                    session, self.trainable_vars)
        # Make sure to close the session, we will make a new one later
        # to save a new checkpoint
        session.close()

    def _make_pruner_spec(self, target, following, source=u''):
        kwargs = {
            "target": target,
            "source": source,
            "following": following}
        pruner_node_spec = FilterPrunerNodeSpec(**kwargs)
        return pruner_node_spec

    def _create_adjacency_list(self, output_node_name):
        adj_list = {}
        already_visited = []
        output_node = self.nodes_map[output_node_name]
        traversal_queue = [output_node]
        while traversal_queue:
            curr_node = traversal_queue.pop(0)
            curr_node_name = graph_utils.node_name_from_input(curr_node.name)
            if curr_node_name not in already_visited:
                already_visited.append(curr_node_name)
                for i, input_node_name in enumerate(curr_node.input):
                    name = graph_utils.node_name_from_input(
                                input_node_name)
                    input_node = self.nodes_map[name]
                    if name not in adj_list: adj_list[name] = []
                    adj_list[name].append(curr_node_name)
                    traversal_queue.append(input_node)
        return adj_list

    def _get_node(self, node_name):
        return self.nodes_map[node_name]

    def _get_neighbours(self, node_name):
        if node_name not in self.neighbors:
            return []
        return self.neighbors[node_name]

    def _get_next_op_instance_name(self, curr_node_name, op_name):
        results = []
        curr_node = self.nodes_map[curr_node_name]
        next_node_names = self._get_neighbours(curr_node_name)
        # Reached the end of graph
        if len(next_node_names) == 0:
            return []
        # Found our node
        if curr_node.op == op_name:
            return [curr_node_name]
        # Need to keep looking
        for next_node_name in next_node_names:
            new_result = self._get_next_op_instance_name(
                                next_node_name, op_name)
            if new_result is not None:
                for nr in new_result:
                    if nr not in results:
                        results.append(nr)
        return results

    def _get_conv_weights_node_name(self, conv_node_name):
        curr_node = self.nodes_map[conv_node_name]
        weights_node_name = graph_utils.remove_ref_from_node_name(
                curr_node.input[1])
        return weights_node_name

    def _get_prune_idxs(self, weights_node_name):
        weights = self.values_map[weights_node_name]
        # plot weights for debug
        if self.interactive_mode:
            plot_magnitude_of_weights(weights_node_name, weights,
                                      self.compression_factor)
        # Find filters to keep
        num_filters = weights.shape[-1]
        num_filters_to_keep = int(num_filters*self.compression_factor)
        weight_magnitudes = np.abs(weights).sum((0,1,2))
        smallest_idxs = np.argsort(-weight_magnitudes)
        top_smallest_idxs = np.sort(smallest_idxs[:num_filters_to_keep])
        prune_idxs = np.zeros(num_filters).astype(bool)
        prune_idxs[top_smallest_idxs] = True
        return prune_idxs

    def _prune_conv_node(self, conv_node_name, idxs=None):
        weights_node_name = self._get_conv_weights_node_name(conv_node_name)
        # If this node has had input channels removed before,
        # we need to reuse that instance of weight values
        if weights_node_name in self.output_values_map:
            weights = self.output_values_map[weights_node_name]
        else:
            weights = self.values_map[weights_node_name]
        # Grab the indices of the weights to prune
        prune_idxs = self._get_prune_idxs(weights_node_name)
        if idxs is not None:
            prune_idxs = idxs
        # Apply by actually changing shape, or simulate pruning
        pruned_weights = np.copy(weights)
        if not self.soft_apply:
            pruned_weights = pruned_weights[:,:,:,prune_idxs]
        else:
            pruned_weights[:,:,:,~prune_idxs] = 0
        # Create new Conv layer with pruned weights
        self.output_values_map[weights_node_name] = pruned_weights
        return weights_node_name, prune_idxs

    def _remove_bn_param_channels(self, next_bn_node_name, prune_idxs):
        curr_node = self.nodes_map[next_bn_node_name]
        # BN param names
        scale_node_name = graph_utils.remove_ref_from_node_name(
                curr_node.input[1])
        shift_node_name = graph_utils.remove_ref_from_node_name(
                curr_node.input[2])
        # Moving average node names
        base_moment_name = next_bn_node_name[:next_bn_node_name.rfind("/")]
        mean_node_name = base_moment_name + "/moving_mean"
        variance_node_name = base_moment_name + "/moving_variance"
        # Adjust values to account for removed idxs
        for var_name in [scale_node_name, shift_node_name,
                         mean_node_name, variance_node_name]:
            value = self.values_map[var_name]
            pruned_value = np.copy(value)
            if not self.soft_apply:
                pruned_value = pruned_value[prune_idxs]
            else:
                pruned_value[~prune_idxs] = 0
            self.output_values_map[var_name] = pruned_value
        return next_bn_node_name

    def _remove_conv_param_channels(self, next_conv_node_name, src_prune_idxs):
        curr_node = self.nodes_map[next_conv_node_name]
        weights_node_name = graph_utils.remove_ref_from_node_name(
                curr_node.input[1])
        # If this node has had FILTERS removed, we can still remove
        # input channels. So this case should be fine
        if weights_node_name in self.output_values_map:
            weights = self.output_values_map[weights_node_name]
        else:
            weights = self.values_map[weights_node_name]
        (kernel_h, kernel_w, batch, num_filters) = weights.shape
        # In order to prune the last Conv Op in the PSPModule, we need
        # to pad the channels of src_prune_idxs to match that output of the
        # Concat op. See the PSP prune config for more information.
        prune_idxs = np.copy(src_prune_idxs)
        if batch != len(src_prune_idxs):
            prune_idxs.resize(batch)
            prune_idxs[len(src_prune_idxs):] = True # keep extra channels
        # Soft apply if we need to retrain without changing variable shape
        updated_kernels = np.copy(weights)
        if not self.soft_apply:
            updated_kernels = updated_kernels[:,:,prune_idxs,:]
        else:
            updated_kernels[:,:,~prune_idxs,:] = 0
        self.output_values_map[weights_node_name] = updated_kernels
        return weights_node_name

    def _apply_pruner_specs(self, pruner_specs):
        self.output_values_map = {}
        pruned_node_idxs = {}
        for pruner_spec in pruner_specs:
            curr_node_name = pruner_spec.target
            source_node_name = pruner_spec.source
            following_node_names = pruner_spec.following
            curr_node = self.nodes_map[curr_node_name]

            print('\x1b[6;30;44m'+
                'Applying Pruning Spec to `%s`!\x1b[0m' % curr_node_name)

            if curr_node.op != "Conv2D":
                raise ValueError("Only Conv nodes can be prunned with the "
                                 "FilterPruner compressor.")
            # Prune the current conv we are dealing with
            source_node_idxs = None
            if source_node_name:
                # TODO(oandrien): This is redundant, should fix traversal
                # instead. Look ahead if we havent encountered the node yet.
                if source_node_name not in pruned_node_idxs:
                    weights_node_name = self._get_conv_weights_node_name(
                        source_node_name)
                    source_node_idxs = self._get_prune_idxs(weights_node_name)
                else:
                    source_node_idxs = pruned_node_idxs[source_node_name]
            # Actually prune the variable now
            new_weights_node_name, prune_idxs = self._prune_conv_node(
                        curr_node_name, source_node_idxs)
            pruned_node_idxs[curr_node_name] = prune_idxs
            # Prune following BN's and Convs
            for following_node_name in following_node_names:
                following_node = self.nodes_map[following_node_name]
                # BATCH NORM
                if following_node.op == "FusedBatchNorm":
                    new_following_node_name = self._remove_bn_param_channels(
                        following_node_name, prune_idxs)
                # CONVS
                elif following_node.op == "Conv2D":
                    new_following_node_name = self._remove_conv_param_channels(
                        following_node_name, prune_idxs)
                else:
                    raise ValueError('Following nodes should only be'
                                     ' BatchNorms or Convs...')
                print('Removed channels from %s...'%new_following_node_name)
        print('\x1b[6;30;42m DONE! \x1b[0m')

    def _get_following_bn_and_conv_names(self, curr_node_name):
        next_conv_node_names = []
        next_bn_node_names = []
        next_node_names = self._get_neighbours(curr_node_name)
        for next_node_name in next_node_names:
            next_node = self.nodes_map[next_node_name]
            if next_node.op == "BatchToSpaceND":
                next_node_name = self._get_next_op_instance_name(
                    next_node_name, "FusedBatchNorm")
                if len(next_node_name) > 1:
                    raise ValueError('Something went wrong with BatchNorms...')
                next_node_name = next_node_name[0]
                next_node = self.nodes_map[next_node_name]
            if next_node.op == "FusedBatchNorm":
                next_bn_node_names = [next_node_name]
                next_conv_node_names = self._get_next_op_instance_name(
                    next_node_name, "Conv2D")
            elif next_node.op == "Relu" or next_node.op == "Relu6":
                next_conv_node_names = self._get_next_op_instance_name(
                    next_node_name, "Conv2D")
            elif next_node.op == "Conv2D":
                next_conv_node_names = [next_node_name]
            elif next_node_name == self.output_node:
                return None
            else:
                raise ValueError('Incompatable model file.')
        return next_bn_node_names + next_conv_node_names

    def _create_pruner_specs_recursively(self, curr_node_name):
         # Update traversal state
        if curr_node_name in self.state.already_visited:
            return
        self.state.already_visited[curr_node_name] = True
        # Get node info
        curr_node = self.nodes_map[curr_node_name]
        next_node_names = self._get_neighbours(curr_node_name)

        if curr_node_name not in self.skippable_nodes:
            # If not conv, we skip since we only deal with
            # convs and djecent convs and proceding batch norms
            if curr_node_name in self.init_pruner_specs.keys():
                pruner_spec = self.init_pruner_specs[curr_node_name]
                self.pruner_specs.append(pruner_spec)

                print('\x1b[6;30;42m'+
                    'Currently on `%s`\x1b[0m' %curr_node_name)
                print(" - Added from INIT_PRUNER_SPEC")
                for name in pruner_spec.following:
                    print(" - Following: " + name)

            elif curr_node.op == "Conv2D":
                # Create filter spec from traversal
                dependant_nodes = self._get_following_bn_and_conv_names(
                        curr_node_name)
                if dependant_nodes is not None and len(dependant_nodes) > 0:
                    pruner_spec = self._make_pruner_spec(
                        curr_node_name, following=dependant_nodes)
                    self.pruner_specs.append(pruner_spec)

                    print('\x1b[6;30;42m'
                        +'Currently on `%s`\x1b[0m' %curr_node_name)
                    print(" - Added from CREATED_PRUNER_SPEC")
                    for name in pruner_spec.following:
                        print(" - Following: " + name)
                else:
                    print('\x1b[6;30;43m'+
                        'Skipping last Conv `%s`\x1b[0m' % curr_node_name)
            else:
                print("Currently on {}, node is Non-Conv, skipping...".format(
                    curr_node_name))
        else:
            print("Currently on {}, node in skip list, skipping...".format(
                curr_node_name))

        # Traverse adjacent nodes
        for next_node in next_node_names:
            self._create_pruner_specs_recursively(next_node)

    def compress(self, input_checkpoint, skip_apply=False):
        # Create a session and graph all variable values
        self._init_pruning_graph(input_checkpoint)
        if self.clear_devices:
            graph_utils.clear_node_devices(self.input_graph_def.node)
        self.neighbors = self._create_adjacency_list(self.output_node)
        # Traverse graph and collect all nodes and dependencies
        self.state = GraphTraversalState(
                already_visited={}, output_node_stack=[])
        self._create_pruner_specs_recursively(self.input_node)
        if not skip_apply:
            self._apply_pruner_specs(self.pruner_specs)

    def save(self, output_checkpoint_dir, output_checkpoint_name):
        output_checkpoint_path = os.path.join(
                output_checkpoint_dir, output_checkpoint_name)
        output_graph = tf.Graph()
        with output_graph.as_default():
            session = tf.Session(graph=output_graph)
            var_list = []
            for trainable_var in self.trainable_vars:
                if (trainable_var in self.skippable_nodes or
                    trainable_var not in self.output_values_map):
                    print('WARNING: Copying original node %s...'%
                        trainable_var)
                    init_value = self.values_map[trainable_var]
                else:
                    init_value = self.output_values_map[trainable_var]
                new_var = graph_utils.add_variable_to_graph(
                    session.graph, trainable_var, init_value)
                var_list.append(new_var)
            # To avoid error with searching for global step
            global_step = tf.train.get_or_create_global_step()
            # Need to run init to assign all our new variable vals
            session.run(tf.variables_initializer(var_list=var_list))
            write_saver = tf.train.Saver(
                var_list=var_list, write_version=self.checkpoint_version)
            write_saver.save(session, output_checkpoint_path)
            print('Saving pruned model checkpoint to {}'.format(
                output_checkpoint_path))
        session.close()
