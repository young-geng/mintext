from typing import Any, Dict, List, Optional, Tuple, Union
import json
from functools import partial
import einops

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
import flax
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from ringattention import blockwise_feedforward, ringattention, ringattention_inference
import mlxu
from scalax.sharding import (
    MeshShardingHelper, TreePathShardingRule, with_sharding_annotation, with_sharding_constraint
)


class LLaMAConfigurator(object):

    @classmethod
    def get_default_config(cls, updates=None):
        config = mlxu.config_dict()
        config.base_model = 'llama3_8b'
        config.vocab_size = mlxu.config_placeholder(int)
        config.hidden_size = mlxu.config_placeholder(int)
        config.intermediate_size = mlxu.config_placeholder(int)
        config.num_hidden_layers = mlxu.config_placeholder(int)
        config.num_attention_heads = mlxu.config_placeholder(int)
        config.num_key_value_heads = mlxu.config_placeholder(int)
        config.initializer_range = mlxu.config_placeholder(float)
        config.rms_norm_eps = mlxu.config_placeholder(float)
        config.seq_length = mlxu.config_placeholder(int)
        config.rope_theta = mlxu.config_placeholder(float)
        config.embedding_dropout = mlxu.config_placeholder(float)
        config.feedforward_dropout = mlxu.config_placeholder(float)
        config.attention_dropout = mlxu.config_placeholder(float)
        config.residue_dropout = mlxu.config_placeholder(float)
        config.scan_attention = mlxu.config_placeholder(bool)
        config.scan_mlp = mlxu.config_placeholder(bool)
        config.scan_layers = mlxu.config_placeholder(bool)
        config.scan_query_chunk_size = mlxu.config_placeholder(int)
        config.scan_key_chunk_size = mlxu.config_placeholder(int)
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def finalize_config(cls, config):
        standard_config = cls.get_standard_llama_config(config.base_model)
        for key, value in config.items():
            if key != 'base_model' and value is not None:
                standard_config[key] = value
        return standard_config

    @classmethod
    def get_standard_llama_config(cls, model_name):
        config = mlxu.config_dict()
        config.base_model = 'llama_7b'
        config.vocab_size = 32000
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.initializer_range = 1.0
        config.rms_norm_eps = 1e-6
        config.seq_length = 2048
        config.rope_theta = 1e4
        config.embedding_dropout = 0.0
        config.feedforward_dropout = 0.0
        config.attention_dropout = 0.0
        config.residue_dropout = 0.0
        config.scan_mlp = False
        config.scan_attention = True
        config.scan_query_chunk_size = 512
        config.scan_key_chunk_size = 512
        config.scan_layers = True
        updates = {
            'debug': dict(
                base_model='debug',
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                rms_norm_eps=1e-6,
            ),
            'llama_0.2b': dict(
                base_model='llama_0.2b',
                hidden_size=1024,
                intermediate_size=2560,
                num_hidden_layers=22,
                num_attention_heads=16,
                num_key_value_heads=16,
                rms_norm_eps=1e-6,
            ),
            'llama_1b': dict(
                base_model='llama_1b',
                hidden_size=2048,
                intermediate_size=5504,
                num_hidden_layers=22,
                num_attention_heads=16,
                num_key_value_heads=16,
                rms_norm_eps=1e-6,
            ),
            'llama_3b': dict(
                base_model='llama_3b',
                hidden_size=3200,
                intermediate_size=8640,
                num_hidden_layers=26,
                num_attention_heads=32,
                num_key_value_heads=32,
                rms_norm_eps=1e-6,
            ),
            'llama_7b': dict(
                base_model='llama_7b',
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                rms_norm_eps=1e-6,
            ),
            'llama_13b': dict(
                base_model='llama_13b',
                hidden_size=5120,
                intermediate_size=13824,
                num_hidden_layers=40,
                num_attention_heads=40,
                num_key_value_heads=40,
                rms_norm_eps=1e-6,
            ),
            'llama_30b': dict(
                base_model='llama_30b',
                hidden_size=6656,
                intermediate_size=17920,
                num_hidden_layers=60,
                num_attention_heads=52,
                num_key_value_heads=52,
                rms_norm_eps=1e-6,
            ),
            'llama_65b': dict(
                base_model='llama_65b',
                hidden_size=8192,
                intermediate_size=22016,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=64,
                rms_norm_eps=1e-5,
            ),
            'llama2_7b': dict(
                base_model='llama2_7b',
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                rms_norm_eps=1e-5,
            ),
            'llama2_13b': dict(
                base_model='llama2_13b',
                hidden_size=5120,
                intermediate_size=13824,
                num_hidden_layers=40,
                num_attention_heads=40,
                num_key_value_heads=40,
                rms_norm_eps=1e-5,
            ),
            'llama2_70b': dict(
                base_model='llama_65b',
                hidden_size=8192,
                intermediate_size=28672,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=8,
                rms_norm_eps=1e-5,
            ),
            'llama3_8b': dict(
                base_model='llama3_8b',
                hidden_size=4096,
                intermediate_size=14336,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                rms_norm_eps=1e-5,
                rope_theta=5e5,
            ),
            'llama3_70b': dict(
                base_model='llama3_8b',
                hidden_size=8192,
                intermediate_size=28672,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=8,
                rms_norm_eps=1e-5,
                rope_theta=5e5,
            ),
        }[model_name]
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def get_model_sharding_rule(cls, scan_layers=False):
        if not scan_layers:
            return TreePathShardingRule(
                # embeddings
                ('transformer/wte/embedding', PS('tp', 'fsdp')),
                # atention
                ('self_attention/(k_proj|q_proj|v_proj)/kernel', PS('fsdp', 'tp')),
                ('self_attention/o_proj/kernel', PS('tp', 'fsdp')),
                # mlp
                ('feedforward/up_proj/kernel', PS('fsdp', 'tp')),
                ('feedforward/down_proj/kernel', PS('tp', 'fsdp')),
                ('feedforward/gate_proj/kernel', PS('fsdp', 'tp')),
                # layer norms
                ('input_layernorm/scale', PS(None)),
                ('post_attention_layernorm/scale', PS(None)),
                # output head
                ('lm_head_norm/scale', PS(None)),
                ('lm_head/kernel', PS('fsdp', 'tp')),
                ('.*', PS(None)),
            )
        else:
            return TreePathShardingRule(
                # embeddings
                ('transformer/wte/embedding', PS('tp', 'fsdp')),
                # atention
                ('self_attention/(k_proj|q_proj|v_proj)/kernel', PS(None, 'fsdp', 'tp')),
                ('self_attention/o_proj/kernel', PS(None, 'tp', 'fsdp')),
                # mlp
                ('feedforward/up_proj/kernel', PS(None, 'fsdp', 'tp')),
                ('feedforward/down_proj/kernel', PS(None, 'tp', 'fsdp')),
                ('feedforward/gate_proj/kernel', PS(None, 'fsdp', 'tp')),
                # layer norms
                ('input_layernorm/scale', PS(None, None)),
                ('post_attention_layernorm/scale', PS(None, None)),
                # output head
                ('lm_head_norm/scale', PS(None)),
                ('lm_head/kernel', PS('fsdp', 'tp')),
                ('.*', PS(None)),
            )

    @classmethod
    def get_jax_mesh(cls, axis_dims):
        if axis_dims.startswith('!'):
            # Allow splitting a physical mesh axis if needed
            mesh_axis_splitting = True
            axis_dims = axis_dims[1:]
        else:
            mesh_axis_splitting = False

        names = ('dp', 'fsdp', 'tp', 'sp')
        dims = [int(x) for x in axis_dims.split(',')]
        assert len(dims) == len(names)
        return MeshShardingHelper(dims, names, mesh_axis_splitting)

    @classmethod
    def rng_keys(cls):
        return ('params', 'dropout')


def apply_rotary_emb(xq, xk, position_ids, max_pos, theta=10000.0):
    input_dtype = xq.dtype
    with jax.ensure_compile_time_eval():
        dim = xq.shape[-1]
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(max_pos)
        freqs = jnp.outer(t, freqs).astype(jnp.float32)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        freqs_cis = jnp.complex64(cos + 1j * sin)
    freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(input_dtype), xk_out.astype(input_dtype)


class FeedForward(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True):
        w1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.hidden_size),
                dtype=self.param_dtype
            ),
            name='gate_proj',
        )
        w2 = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.intermediate_size),
                dtype=self.param_dtype
            ),
            name='down_proj',
        )
        w3 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.hidden_size),
                dtype=self.param_dtype
            ),
            name='up_proj',
        )
        x = w2(
            nn.silu(with_sharding_annotation(w1(x), 'ffw_intermediate'))
            * with_sharding_annotation(w3(x), 'ffw_intermediate')
        )
        x = with_sharding_annotation(x, 'ffw_output')
        return nn.Dropout(rate=self.config.feedforward_dropout)(x, deterministic=deterministic)


class Attention(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def _concatenate_to_cache(self, key, value, query):
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            if query.shape[1] == 1:
                mesh = MeshShardingHelper.get_global_mesh()
                def fn(cached_key, cached_value, key, value, cur_index):
                    assert key.shape[1] == 1 and value.shape[1] == 1, (key.shape, value.shape)
                    sp_size = max_length // mesh.shape['sp']
                    axis_index = jax.lax.axis_index('sp')
                    cur_index = cur_index - axis_index * sp_size
                    key, value = jax.lax.cond(
                        jnp.logical_and(cur_index >= 0, cur_index < sp_size),
                        lambda: (
                            cached_key.at[:, cur_index].set(key[:, -1]),
                            cached_value.at[:, cur_index].set(value[:, -1]),
                        ),
                        lambda: (cached_key, cached_value),
                    )
                    return key, value
                fn = shard_map(
                    fn, mesh=mesh,
                    in_specs=(
                        PS(('dp', 'fsdp'), 'sp', 'tp', None),
                        PS(('dp', 'fsdp'), 'sp', 'tp', None),
                        PS(('dp', 'fsdp'), None, 'tp', None),
                        PS(('dp', 'fsdp'), None, 'tp', None),
                        PS()
                    ),
                    out_specs=(
                        PS(('dp', 'fsdp'), 'sp', 'tp', None),
                        PS(('dp', 'fsdp'), 'sp', 'tp', None)
                    ),
                    check_rep=False
                )
                key, value = fn(cached_key.value, cached_value.value, key, value, cur_index)
            else:
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
        return key, value

    @nn.compact
    def __call__(self, hidden_states, attention_mask, segment_ids, position_ids, deterministic, init_cache):
        assert self.config.hidden_size % self.config.num_key_value_heads == 0
        assert self.config.hidden_size % self.config.num_attention_heads == 0

        init_scale = self.config.initializer_range / np.sqrt(self.config.hidden_size)
        num_query_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        xq = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='q_proj',
        )(hidden_states)
        xk = nn.Dense(
            self.config.hidden_size // num_query_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='k_proj',
        )(hidden_states)
        xv = nn.Dense(
            self.config.hidden_size // num_query_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='v_proj',
        )(hidden_states)

        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), "sp", "tp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), "sp", "tp"))
        if xq.shape[1] == 1:
            xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "tp"))
        else:
            xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), "sp", "tp"))

        xq = einops.rearrange(
            xq, 'b s (h d) -> b s h d',
            h=self.config.num_attention_heads,
        )
        xk = einops.repeat(
            xk, 'b s (h d) -> b s (h g) d',
            h=self.config.num_key_value_heads,
            g=num_query_groups,
        )
        xv = einops.repeat(
            xv, 'b s (h d) -> b s (h g) d',
            h=self.config.num_key_value_heads,
            g=num_query_groups,
        )

        if position_ids is None:
            position_ids = jnp.arange(0, self.config.seq_length, dtype=jnp.int32)[None].repeat(hidden_states.shape[0], axis=0)
        xq, xk = apply_rotary_emb(
            xq, xk, position_ids,
            max_pos=self.config.seq_length,
            theta=self.config.rope_theta,
        )

        if self.config.scan_attention and xq.shape[1] > max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size):
            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv = self._concatenate_to_cache(xk, xv, xq)

            if attention_mask is not None:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
                attention_bias = lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
                )
            else:
                attention_bias = None
            ring_attention_sharded = shard_map(
                partial(
                    ringattention,
                    axis_name="sp",
                    float32_logits=True,
                    cache_idx=None,
                    blockwise_kwargs=dict(
                        causal_block_size=1,
                        deterministic=deterministic,
                        dropout_rng=None,
                        attn_pdrop=0,
                        query_chunk_size=self.config.scan_query_chunk_size,
                        key_chunk_size=self.config.scan_key_chunk_size,
                        dtype=self.dtype,
                        policy=jax.checkpoint_policies.nothing_saveable,
                        precision=None,
                        prevent_cse=not self.config.scan_layers,
                    )
                ),
                mesh=MeshShardingHelper.get_global_mesh(),
                in_specs=(
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), None, None, None),
                    PS(("dp", "fsdp"), None),
                ),
                out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
                check_rep=False
            )
            attention_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids)
            attention_output = with_sharding_constraint(attention_output, PS(("dp", "fsdp"), "sp", "tp", None))
        else:
            query_length, key_length = xq.shape[1], xk.shape[1]

            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = jnp.arange(max_decoder_length)[None] <= (jnp.arange(query_length) + mask_shift)[:, None]
                causal_mask = causal_mask[None, None]
                segment_mask = None
            else:
                causal_mask = nn.attention.make_causal_mask(jnp.ones((1, self.config.seq_length), dtype="bool"), dtype="bool")
                causal_mask = causal_mask[:, :, :query_length, :key_length]
                if segment_ids is not None:
                    segment_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
                    segment_mask = segment_mask[:, None]
                else:
                    segment_mask = None

            batch_size = hidden_states.shape[0]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

            if attention_mask is not None:
                attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = nn.attention.combine_masks(attention_mask, causal_mask, segment_mask)

            if self.has_variable("cache", "cached_key") or init_cache:
                xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)

            q_sp_dim = None if xq.shape[1] == 1 else 'sp'
            ring_attention_sharded = shard_map(
                partial(ringattention_inference, axis_name="sp"), mesh=MeshShardingHelper.get_global_mesh(),
                in_specs=(
                    PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), None, q_sp_dim, None)
                ),
                out_specs=PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                check_rep=False
            )
            attention_output = ring_attention_sharded(
                xq, xk, xv, attention_mask
            )

        attention_output = einops.rearrange(attention_output, 'b s h d -> b s (h d)')

        x_out = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='o_proj',
        )(attention_output)

        x_out = nn.Dropout(rate=self.config.residue_dropout)(
            x_out, deterministic=deterministic
        )
        return x_out


class TransformerBlock(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, attention_mask, segment_ids, position_ids, deterministic, init_cache):
        x_out = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            param_dtype=self.param_dtype,
            name='input_layernorm',
        )(hidden_states)
        x_out = Attention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='self_attention',
        )(x_out, attention_mask, segment_ids, position_ids, deterministic, init_cache)
        mlp_inputs = x_out + hidden_states
        x_out = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            param_dtype=self.param_dtype,
            name='post_attention_layernorm',
        )(mlp_inputs)
        if self.config.scan_mlp:
            feed_forward_module = nn.partitioning.remat(
                FeedForward, static_argnums=(1,),
                policy=jax.checkpoint_policies.nothing_saveable,
                prevent_cse=not self.config.scan_layers,
            )
        else:
            feed_forward_module = FeedForward
        feed_forward = feed_forward_module(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='feedforward',
        )
        if self.config.scan_mlp and hidden_states.shape[1] >= self.config.scan_mlp_chunk_size:
            x_out = blockwise_feedforward(feed_forward, x_out, self.config.scan_mlp_chunk_size, pre_remat=True)
        else:
            x_out = feed_forward(x_out, deterministic)
        x_out = with_sharding_constraint(x_out, PS(("dp", "fsdp"), None, "tp"))
        outputs = x_out + mlp_inputs
        if self.config.scan_layers:
            return (outputs, None)
        return outputs


class LLaMAModel(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, segment_ids=None, position_ids=None, deterministic=True, init_cache=False):
        hidden_states = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                self.config.initializer_range,
                dtype=self.param_dtype
            ),
            dtype=self.param_dtype,
            name='embeddings',
        )(input_ids)
        hidden_states = nn.Dropout(
            rate=self.config.embedding_dropout, name='emb_drop'
        )(hidden_states, deterministic=deterministic)

        if self.config.scan_layers:
            initializing = self.is_mutable_collection('params')
            params_spec = (
                0 if initializing else
                nn_partitioning.ScanIn(0))
            cache_spec = 0
            hidden_states, _ = nn.scan(
                TransformerBlock,
                variable_axes={
                    'params': params_spec,
                    'cache': cache_spec,
                    'intermediates': 0
                },
                split_rngs={
                    'params': True,
                    'dropout': True
                },
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
                length=self.config.num_hidden_layers,
                metadata_params={nn.PARTITION_NAME: 'scan_decoder_layer'},
                )(self.config, name='scan_decoder', dtype=self.dtype, param_dtype=self.param_dtype)(
                    hidden_states,
                    attention_mask,
                    segment_ids,
                    position_ids,
                    deterministic,
                    init_cache,
                )
        else:
            blocks = [
                TransformerBlock(
                    self.config,
                    name=f'transformer_block_{i}',
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ) for i in range(self.config.num_hidden_layers)
            ]
            for block in blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    segment_ids,
                    position_ids,
                    deterministic,
                    init_cache,
                )
        hidden_states = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            name='lm_head_norm',
        )(hidden_states)
        logits = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.hidden_size),
                dtype=self.param_dtype
            ),
            name='lm_head',
        )(hidden_states)
        return logits
