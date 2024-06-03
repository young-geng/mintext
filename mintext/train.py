import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from scalax.sharding import with_sharding_constraint, with_sharding_annotation
from scalax.utils import JaxRNG, get_float_dtype_by_name
from flax.training.train_state import TrainState
from transformers import AutoTokenizer

from mintext.data import JsonDataset
from mintext.utils import (
    JaxDistributedConfigurator, AdamConfigurator, Checkpointer,
    global_norm, cross_entropy_loss_and_accuracy, average_metrics,
)
from mintext.model import LLaMAConfigurator, LLaMAModel


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    param_dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_params_checkpoint='',
    load_train_state_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset=JsonDataset.get_default_config(),
    eval_dataset=JsonDataset.get_default_config(),
    optimizer=AdamConfigurator.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    checkpointer=Checkpointer.get_default_config(),
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
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = JsonDataset(FLAGS.eval_dataset, tokenizer)
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

    model = LLaMAModel(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    optimizer, lr_schedule = AdamConfigurator.get_optimizer_and_schedule(
        FLAGS.optimizer
    )
    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    checkpointer = Checkpointer(FLAGS.checkpointer)


    @partial(
        mesh.sjit,
        in_shardings=None,
        out_shardings=LLaMAConfigurator.get_model_sharding_rule(),
        annotation_shardings=LLaMAConfigurator.get_intermediate_sharding_rules(),
    )
    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    @partial(
        mesh.sjit,
        in_shardings=(
            LLaMAConfigurator.get_model_sharding_rule(),
            PS(),
            PS(),
        ),
        out_shardings=(
            LLaMAConfigurator.get_model_sharding_rule(),
            PS(),
            PS(),
        ),
        args_sharding_constraint=(
            LLaMAConfigurator.get_model_sharding_rule(),
            PS(),
            PS(('replica', 'fsdp')),
        ),
        annotation_shardings=LLaMAConfigurator.get_intermediate_sharding_rules(),
        donate_argnums=(0, ),
    )
    def train_step_fn(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        def loss_and_accuracy(params):
            logits = model.apply(
                params,
                input_ids=batch['input_tokens'],
                attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'],
                deterministic=False,
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            )
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=lr_schedule(train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    @partial(
        mesh.sjit,
        in_shardings=(
            LLaMAConfigurator.get_model_sharding_rule(),
            PS(),
            PS(),
        ),
        out_shardings=(PS(), PS()),
        args_sharding_constraint=(
            LLaMAConfigurator.get_model_sharding_rule(),
            PS(),
            PS(('replica', 'fsdp')),
        ),
        annotation_shardings=LLaMAConfigurator.get_intermediate_sharding_rules(),
    )
    def eval_step_fn(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        logits = model.apply(
            train_state.params,
            input_ids=batch['input_tokens'],
            attention_mask=batch['attention_masks'],
            position_ids=batch['position_ids'],
            deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
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
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save(
            train_state=train_state,
            metadata=metadata,
            dataset_state=dataset.get_state_dict(),
            prefix=f'step_{step}' if milestone else 'latest',
        )

    train_state = init_fn(JaxRNG.next_rng())

    if FLAGS.load_train_state_checkpoint != '':
        train_state_shapes = checkpointer.get_shape_dtype_struct(train_state)
        train_state = None  # Release memory before loading
        train_state = checkpointer.restore(
            FLAGS.load_train_state_checkpoint, train_state_shapes
        )
    elif FLAGS.load_params_checkpoint != '':
        model_shapes = checkpointer.get_shape_dtype_struct(train_state.params)
        train_state = train_state.replace(params=None)  # Release memory before loading
        train_state = train_state.replace(
            params=checkpointer.restore(
                FLAGS.load_params_checkpoint, model_shapes
            )
        )

    start_step = int(jax.device_get(train_state.step))

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