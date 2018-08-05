import os
import functools

from protos import compressor_pb2
from libs.filter_pruner import FilterPruner, FilterPrunerNodeSpec

def _complete_node_scope(name, parent_scope, overide_scope=None):
    if not name:
        return ''
    if name[:3] == "...":
        return name[3:]
    main_scope = overide_scope if overide_scope is not None else parent_scope
    return os.path.join(main_scope, name)

def _build_filter_pruning_compressor(filter_pruning_config, skippable_nodes,
                                     compression_factor, interactive_mode,
                                     soft_apply):
    input_node_name = filter_pruning_config.input.name
    output_node_name = filter_pruning_config.output.name
    # skippable_nodes
    config_skip_nodes = []
    for skip_node in filter_pruning_config.skip_node:
        config_skip_nodes.append(skip_node.name)
    config_skip_nodes += skippable_nodes
    # Get prespecified pruner specs for complex nodes pruning schemes
    pruner_specs = {}
    nonoveride_complete_scope = functools.partial(
        _complete_node_scope,
        parent_scope=filter_pruning_config.node_scope)
    for node in filter_pruning_config.node:
        overide_scope = (node.node_scope
            if node.node_scope != "null" else None)
        complete_scope = functools.partial(
            nonoveride_complete_scope,
            overide_scope=overide_scope)
        pruner_spec_key = complete_scope(node.target.name)
        following = []
        for follow_node in node.following:
            following.append(complete_scope(follow_node.name))
        # import pdb; pdb.set_trace()
        pruner_spec = FilterPrunerNodeSpec(
            source=complete_scope(node.source.name),
            target=complete_scope(node.target.name),
            following=following)
        pruner_specs[pruner_spec_key] = pruner_spec
    # Pruner class to use in script
    pruner = FilterPruner(input_node=input_node_name,
                          output_node=output_node_name,
                          compression_factor=compression_factor,
                          init_pruner_specs=pruner_specs,
                          skippable_nodes=config_skip_nodes,
                          interactive_mode=interactive_mode,
                          soft_apply=soft_apply)
    return pruner


def build(compression_config, skippable_nodes, compression_factor,
          interactive_mode=False, soft_apply=False):
    if not isinstance(compression_config, compressor_pb2.CompressionStrategy):
        raise ValueError('pruner_config not of type '
                         'compressor_pb2.CompressionStrategy.')
    compression_strategy = compression_config.WhichOneof('compression_strategy')
    if compression_strategy == 'filter_pruner':
        return _build_filter_pruning_compressor(
                    compression_config.filter_pruner,
                    skippable_nodes,
                    compression_factor,
                    interactive_mode,
                    soft_apply)

    raise ValueError('Unknown compression strategy: {}'.format(
        compression_strategy))
