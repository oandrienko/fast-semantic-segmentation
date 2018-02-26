import tensorflow as tf


def _create_learning_rate(learning_rate_config):
    learning_rate = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')

    # Authors of ENet claim to use constant constant learning rate policy
    # with a constant value of of 0.0002
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        learning_rate = tf.constant(config.learning_rate, dtype=tf.float32)

    # For ICNet, authors claim to use "poly" learning rate policy with a
    # base learning rate is 0.01 and with power of 0.9 for growth
    if learning_rate_type == 'polynomial_decay_learning_rate':
        config = config = learning_rate_config.polynomial_decay_learning_rate
        learning_rate = tf.train.polynomial_decay(
            config.initial_learning_rate,
            tf.train.get_or_create_global_step(),
            config.decay_steps,
            power=config.power,
            end_learning_rate=0)

    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        learning_rate = tf.train.exponential_decay(
            config.initial_learning_rate,
            tf.train.get_or_create_global_step(),
            config.decay_steps,
            config.decay_factor,
            staircase=config.staircase)

    if learning_rate is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

    return learning_rate


def build(optimizer_config):
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None

    summary_vars = []

    # Used by authors of ICNet for training. Momentum is set to 0.9.
    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=config.momentum_optimizer_value)

    # The ADAM optimizer is used by ENet authors
    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    return optimizer, summary_vars
