from functools import partial
import os
import mlxu

import numpy as np
import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from scalax.sharding import MeshShardingHelper


class JaxDistributedConfigurator(object):
    """ Utility class for initializing JAX distributed. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.initialize_jax_distributed = False
        config.coordinator_address = mlxu.config_placeholder(str)
        config.num_processes = mlxu.config_placeholder(int)
        config.process_id = mlxu.config_placeholder(int)
        config.local_device_ids = mlxu.config_placeholder(str)
        return mlxu.update_config_dict(config, updates)


    @classmethod
    def initialize(cls, config):
        config = cls.get_default_config(config)
        if config.initialize_jax_distributed:
            if config.local_device_ids is not None:
                local_device_ids = [int(x) for x in config.local_device_ids.split(',')]
            else:
                local_device_ids = None

            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=config.process_id,
                local_device_ids=local_device_ids,
            )


class Checkpointer(object):
    """ A simple wrapper for orbax checkpointing. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self.checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler(
                'train_state', 'dataset_state', 'metadata'
            )
        )

    def save(self, train_state, dataset_state=None, metadata=None, prefix=None):
        if self.config.path == '':
            return
        composite_args = {}
        composite_args['train_state'] = ocp.args.StandardSave(train_state)
        if dataset_state is not None:
            composite_args['dataset_state'] = ocp.args.JsonSave(dataset_state)
        if metadata is not None:
            composite_args['metadata'] = ocp.args.JsonSave(metadata)

        if prefix is None:
            path = self.config.path
        else:
            path = os.path.join(self.config.path, prefix)
        self.checkpointer.save(path, args=ocp.args.Composite(**composite_args))

    @classmethod
    def restore(cls, path, item):
        return ocp.StandardCheckpointer().restore(
            path, args=ocp.args.StandardRestore(item)
        )

    @classmethod
    def restore_metadata(cls, path):
        return ocp.StandardCheckpointer().restore(
            path, args=ocp.args.JsonRestore()
        )

    @classmethod
    def get_shape_dtype_struct(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)



class AdamConfigurator(object):
    """ AdamW optimizer with cosine schedule. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.init_lr = 0.0
        config.end_lr = 0.001
        config.lr = 0.01
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 500000
        config.b1 = 0.9
        config.b2 = 0.95
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def get_optimizer_and_schedule(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_decay_steps,
            end_value=config.end_lr,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=config.weight_decay,
                b1=config.b1,
                b2=config.b2,
                mask=weight_decay_mask,
            ),
        )
        return optimizer, learning_rate_schedule


def get_metrics(metrics, unreplicate=False, stack=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    if stack:
        return jax.tree_map(lambda *args: np.stack(args), *metrics)
    else:
        return {key: float(val) for key, val in metrics.items()}


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32) # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )
