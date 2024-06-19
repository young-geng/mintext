import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import mlxu
import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
from scalax.utils import JaxRNG, get_float_dtype_by_name
import torch
import einops
from transformers import AutoModelForCausalLM
from mintext.model import LLaMAConfigurator, LLaMAModel
from mintext.utils import match_and_transform_dict


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    hf_pretrained='',
    output_path='',
    param_dtype=('fp32', 'dtype to save the jax parameters in'),
    torch_dtype=('fp16', 'dtype to load the torch parameters in'),
    llama=LLaMAConfigurator.get_default_config(),
)


def get_torch_dtype_by_name(dtype):
    return {
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[dtype]


def main(argv):
    JaxRNG.init_global_rng(42)
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)
    param_dtype = get_float_dtype_by_name(FLAGS.param_dtype)
    model = LLaMAModel(
        llama_config,
        param_dtype=param_dtype,
    )

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        seq_length = llama_config.max_position_embeddings
        return model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )

    flattend_params_struct = flax.traverse_util.flatten_dict(
        jax.eval_shape(init_fn, JaxRNG.next_rng()), sep='/',
    )

    hf_weights = AutoModelForCausalLM.from_pretrained(
        FLAGS.hf_pretrained,
        torch_dtype=get_torch_dtype_by_name(FLAGS.torch_dtype),
    ).state_dict()

    def to_jnp(x):
        return jnp.array(x.cpu().numpy()).astype(param_dtype)

    def to_jnp_transpose(x):
        return jnp.array(x.cpu().numpy()).astype(param_dtype).T

    def to_jnp_q_proj(x):
        x = einops.rearrange(
            x,
            '(h p d) di -> di (h d p)',
            h=llama_config.num_attention_heads,
            p=2,
        )
        return to_jnp(x)

    def to_jnp_k_proj(x):
        x = einops.rearrange(
            x,
            '(h p d) di -> di (h d p)',
            h=llama_config.num_key_value_heads,
            p=2,
        )
        return to_jnp(x)

    rules = [
        (r'model.embed_tokens.weight',
         r'params/embeddings/embedding',
         to_jnp),
        (r'lm_head.weight',
         r'params/lm_head/kernel',
         to_jnp_transpose),
        (r'model.norm.weight',
         r'params/lm_head_norm/scale',
         to_jnp),
        (r'model.layers.(\d+).self_attn.k_proj.weight',
         r'params/transformer_block_\1/self_attention/k_proj/kernel',
         to_jnp_k_proj),
        (r'model.layers.(\d+).self_attn.o_proj.weight',
         r'params/transformer_block_\1/self_attention/o_proj/kernel',
         to_jnp_transpose),
        (r'model.layers.(\d+).self_attn.q_proj.weight',
         r'params/transformer_block_\1/self_attention/q_proj/kernel',
         to_jnp_q_proj),
        (r'model.layers.(\d+).self_attn.v_proj.weight',
         r'params/transformer_block_\1/self_attention/v_proj/kernel',
         to_jnp_transpose),
        (r'model.layers.(\d+).mlp.down_proj.weight',
         r'params/transformer_block_\1/feedforward/down_proj/kernel',
         to_jnp_transpose),
        (r'model.layers.(\d+).mlp.gate_proj.weight',
         r'params/transformer_block_\1/feedforward/gate_proj/kernel',
         to_jnp_transpose),
        (r'model.layers.(\d+).mlp.up_proj.weight',
         r'params/transformer_block_\1/feedforward/up_proj/kernel',
         to_jnp_transpose),
        (r'model.layers.(\d+).input_layernorm.weight',
         r'params/transformer_block_\1/input_layernorm/scale',
         to_jnp),
        (r'model.layers.(\d+).post_attention_layernorm.weight',
         r'params/transformer_block_\1/post_attention_layernorm/scale',
         to_jnp),
    ]
    converted_params = match_and_transform_dict(hf_weights, rules, donate=True)
    assert len(flattend_params_struct) == len(converted_params)
    for key in flattend_params_struct:
        assert key in converted_params, 'Key not found: {}'.format(key)

    params = flax.traverse_util.unflatten_dict(converted_params, sep='/')
    del converted_params, hf_weights
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(FLAGS.output_path, params, force=True)
    mlxu.open_file(
        os.path.join(FLAGS.output_path, 'commit_success.txt'), 'w'
    ).close()


if __name__ == '__main__':
    mlxu.run(main)