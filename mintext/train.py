import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
import optax
from scalax.utils import JaxRNG, get_float_dtype_by_name
from flax.training.train_state import TrainState
from transformers import AutoTokenizer

from mintext.data import JsonDataset
from mintext.utils import (
    JaxDistributedConfigurator, AdamConfigurator, Checkpointer,
    global_norm, cross_entropy_loss_and_accuracy, average_metrics,
)
from mintext.models import ModelConfigurator


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    dtype='fp32',
    param_dtype='fp32',
    total_steps=10000,
    load_params_checkpoint='',
    load_train_state_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer='openlm-research/open_llama_3b_v2',
    checkpoint_path='',
    checkpoint_separate_params=True,
    train_dataset=JsonDataset.get_default_config(),
    eval_dataset=JsonDataset.get_default_config(),
    optimizer=AdamConfigurator.get_default_config(),
    model=ModelConfigurator.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfigurator.get_default_config(),
)


def main(argv):
    JaxDistributedConfigurator.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    JaxRNG.init_global_rng(FLAGS.seed)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = JsonDataset(FLAGS.train_dataset, tokenizer)

    if FLAGS.eval_steps > 0:
        eval_dataset = JsonDataset(FLAGS.eval_dataset, tokenizer)
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    model_config, model, model_sharding = ModelConfigurator.make_model_and_sharding(
        config=FLAGS.model,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )
    optimizer, lr_schedule = AdamConfigurator.get_optimizer_and_schedule(
        FLAGS.optimizer
    )
    mesh = model_sharding.get_mesh()

    if FLAGS.checkpoint_path == '':
        FLAGS.checkpoint_path = logger.output_dir
    checkpointer = Checkpointer(FLAGS.checkpoint_path)

    @partial(
        mesh.sjit,
        in_shardings=None,
        out_shardings=model_sharding.get_model_sharding_rule(),
        annotation_shardings=model_sharding.get_intermediate_sharding_rules(),
    )
    def init_fn(rng):
        params = model.init(rng)
        opt_state = optimizer.init(params)
        return {
            'params': params,
            'opt_state': opt_state,
            'step': jnp.array(0, jnp.int32)
        }

    @partial(
        mesh.sjit,
        in_shardings=(
            model_sharding.get_model_sharding_rule(),
            PS(),
            PS(),
        ),
        out_shardings=(
            model_sharding.get_model_sharding_rule(),
            PS(),
            PS(),
        ),
        args_sharding_constraint=(
            model_sharding.get_model_sharding_rule(),
            PS(),
            model_sharding.get_batch_sharding(),
        ),
        annotation_shardings=model_sharding.get_intermediate_sharding_rules(),
        donate_argnums=(0, ),
    )
    def train_step_fn(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        def loss_and_accuracy(params):
            logits = model.forward(
                params,
                input_ids=batch['input_tokens'],
                attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'],
                segment_ids=batch['segment_ids'],
            )
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state['params'])
        updates, train_state['opt_state'] = optimizer.update(
            grads, train_state['opt_state'], train_state['params']
        )
        train_state['params'] = optax.apply_updates(train_state['params'], updates)
        train_state['step'] += 1
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=lr_schedule(train_state['step']),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state['params']),
        )
        return train_state, rng_generator(), metrics

    @partial(
        mesh.sjit,
        in_shardings=(
            model_sharding.get_model_sharding_rule(),
            PS(),
            PS(),
        ),
        out_shardings=(PS(), PS()),
        args_sharding_constraint=(
            model_sharding.get_model_sharding_rule(),
            PS(),
            model_sharding.get_batch_sharding(),
        ),
        annotation_shardings=model_sharding.get_intermediate_sharding_rules(),
    )
    def eval_step_fn(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        logits = model.forward(
            train_state['params'],
            input_ids=batch['input_tokens'],
            attention_mask=batch['attention_masks'],
            position_ids=batch['position_ids'],
            segment_ids=batch['segment_ids'],
        )
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits,batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state['step']))
        checkpoint_name = f'step_{step}' if milestone else 'latest'
        # Save the main train state
        checkpointer.save_pytree(
            train_state, prefix=f'train_state_{checkpoint_name}',
        )
        if FLAGS.checkpoint_separate_params:
            # Optionally save the model parameters separately
            checkpointer.save_pytree(
                train_state['params'], prefix=f'params_{checkpoint_name}',
            )
        # Save dataset state and training configs
        checkpointer.save_json(
            dataset.get_state_dict(), f'dataset_state_{checkpoint_name}.json',
        )
        checkpointer.save_json(flags_config_dict.to_dict(), 'flags.json')
        checkpointer.save_json(model_config.to_dict(), 'model_config.json')

    train_state = init_fn(JaxRNG.next_rng())

    if FLAGS.load_train_state_checkpoint != '':
        # Loading a full train state with model params and optimizer state
        assert FLAGS.load_params_checkpoint == '', 'Cannot load both train state and params.'
        train_state_shapes = checkpointer.get_shape_dtype_struct(train_state)
        train_state = None  # Release memory before loading
        train_state = checkpointer.restore_pytree(
            FLAGS.load_train_state_checkpoint, train_state_shapes
        )
    elif FLAGS.load_params_checkpoint != '':
        # Loading only the model parameters
        model_shapes = checkpointer.get_shape_dtype_struct(train_state['params'])
        train_state['params'] = None  # Release memory before loading
        train_state['params'] = checkpointer.restore_pytree(
            FLAGS.load_params_checkpoint, model_shapes
        )

    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(checkpointer.load_json(FLAGS.load_dataset_state))

    start_step = int(jax.device_get(train_state['step']))

    if FLAGS.save_model_freq > 0:
        save_checkpoint(train_state)

    sharded_rng = JaxRNG.next_rng()

    step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

    for step, (batch, dataset_metrics) in zip(step_counter, dataset):
        train_state, sharded_rng, metrics = train_step_fn(
            train_state, sharded_rng, batch
        )

        if step % FLAGS.log_freq == 0:
            if FLAGS.eval_steps > 0:
                eval_metric_list = []
                for _ in range(FLAGS.eval_steps):
                    eval_batch, _ = next(eval_iterator)
                    sharded_rng, eval_metrics = eval_step_fn(
                        train_state, sharded_rng, eval_batch
                    )
                    eval_metric_list.append(eval_metrics)
                metrics.update(average_metrics(eval_metric_list))

            log_metrics = {"step": step}
            log_metrics.update(metrics)
            log_metrics.update(dataset_metrics)
            log_metrics = jax.device_get(log_metrics)
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

        if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
            save_checkpoint(train_state, milestone=True)
        elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
            save_checkpoint(train_state)

    if FLAGS.save_model_freq > 0:
        save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)