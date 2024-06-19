from functools import partial
import os
import re
import json
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


class Checkpointer(object):
    """ A simple wrapper for orbax checkpointing. """

    def __init__(self, path):
        self.path = path
        self.checkpointer = ocp.StandardCheckpointer()
        if self.path != '':
            mlxu.makedirs(self.path)

    def save_pytree(self, pytree, prefix=None):
        """ Save pytree of JAX arrays. """
        if self.path == '':
            return
        if prefix is None:
            path = self.path
        else:
            path = os.path.join(self.path, prefix)

        self.checkpointer.save(path, pytree, force=True)
        # Create a commit_success.txt file to indicate that the checkpoint is
        # saved successfully. This is a workaround for orbax so that locally
        # saved checkpoint can be restored when copied to Google cloud storage.
        mlxu.open_file(os.path.join(path, 'commit_success.txt'), 'w').close()

    @classmethod
    def restore_pytree(cls, path, item):
        return ocp.StandardCheckpointer().restore(
            path, args=ocp.args.StandardRestore(item)
        )

    def save_json(self, data, name):
        """ Save dictionary as JSON. """
        if self.path == '':
            return
        path = os.path.join(self.path, name)
        mlxu.makedirs(path)
        with mlxu.open_file(path, 'w') as f:
            f.write(json.dumps(data, indent=4))

    @classmethod
    def load_json(cls, path):
        with mlxu.open_file(path, 'r') as f:
            return json.loads(f.read())

    @classmethod
    def get_shape_dtype_struct(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)


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


def match_and_transform_dict(data, rules, strict=True, donate=False):
    """ Match keys in a dictionary to a set of rules and transform the value. """
    transformed = {}
    for key in list(data.keys()):
        matched = False
        for src_pattern, tgt_pattern, transform_fn in rules:
            if re.match(src_pattern, key) is not None:
                matched = True
                target_key = re.sub(src_pattern, tgt_pattern, key)
                if transform_fn is None:
                    transformed[target_key] = data[key]
                else:
                    transformed[target_key] = transform_fn(data[key])

                if donate:
                    del data[key]

                break
        if strict and not matched:
            raise ValueError(f"Key {key} does not match any pattern in rules")
    return transformed
