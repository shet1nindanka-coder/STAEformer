import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchinfo import summary


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    # CUDA caps gridDim.y / gridDim.z at 65535, and the fused SDPA kernels map the
    # leading batch dimension onto one of them. Stay safely below the limit.
    MAX_FUSED_BATCH = 32768

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def _split_heads(self, tensor, length):
        """(..., length, model_dim) -> (prod(leading_dims), num_heads, length, head_dim)

        Heads are taken as contiguous slices of the last axis, so head `h` owns
        channels [h * head_dim : (h + 1) * head_dim] -- identical to the previous
        `torch.split(t, head_dim, dim=-1)` implementation.
        """
        tensor = tensor.reshape(-1, length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        leading_shape = query.shape[:-2]  # (batch_size, ...)
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # (..., length, model_dim) -> (batch, num_heads, length, head_dim)
        query = self._split_heads(query, tgt_length)
        key = self._split_heads(key, src_length)
        value = self._split_heads(value, src_length)

        # The fast SDPA kernels (flash / mem-efficient) require head_dim to be a
        # multiple of 8. STAEformer's default is 152/4 = 38, which disqualifies them
        # and silently falls back to the `math` backend -- which materialises the
        # full (..., N, N) score matrix and defeats the whole point.
        #
        # Zero-padding head_dim up to the next multiple of 8 is exact:
        #   * padded Q/K contribute 0 to every dot product -> scores unchanged;
        #   * padded V produces zero output channels, which we slice off.
        # `scale` must stay tied to the TRUE head_dim, not the padded one.
        pad = (-self.head_dim) % 8
        if pad:
            query = F.pad(query, (0, pad))
            key = F.pad(key, (0, pad))
            value = F.pad(value, (0, pad))

        # Fused attention: the (..., tgt_length, src_length) score matrix is never
        # materialised, so memory is linear in sequence length instead of quadratic.
        # Dropout stays 0.0 here: STAEformer applies dropout *after* attention
        # (SelfAttentionLayer.dropout1), not on the attention weights.
        #
        # The fused kernels map the leading (batch) dimension onto a CUDA grid axis
        # capped at 65535. Temporal attention folds num_nodes into that dimension
        # (batch * num_nodes), which overflows on large graphs -- e.g. CA with
        # 8600 nodes at batch 8 gives 68800 and fails with
        # `CUDA error: invalid configuration argument`. Chunking is exact: samples
        # in the leading dimension never attend to each other.
        chunks = [
            F.scaled_dot_product_attention(
                query[start : start + self.MAX_FUSED_BATCH],
                key[start : start + self.MAX_FUSED_BATCH],
                value[start : start + self.MAX_FUSED_BATCH],
                dropout_p=0.0,
                is_causal=self.mask,
                scale=self.head_dim**-0.5,
            )
            for start in range(0, query.shape[0], self.MAX_FUSED_BATCH)
        ]
        # Один чанк -- обычный случай (всё, кроме temporal на CA и breadth на
        # GBA). torch.cat даже из одного тензора аллоцирует и копирует полный
        # выход внимания, а это сотни мегабайт на вызов.
        out = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

        if pad:
            out = out[..., : self.head_dim]

        # (batch, num_heads, tgt_length, head_dim) -> (..., tgt_length, model_dim)
        out = out.transpose(1, 2).reshape(*leading_shape, tgt_length, self.model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        patch_index=None,
        use_checkpoint=False,
    ):
        """
        patch_index : словарь из lib.spatial_patching, либо None.
            Если задан, плотное внимание по узлам заменяется на пару
            depth/breadth по патчам: стоимость падает с O(N^2) до O(M*(P+R)).
            Это ДРУГАЯ модель, а не ускоренная версия прежней -- меняется то,
            какие узлы видят друг друга, поэтому качество надо перемерять.
        use_checkpoint : пересчитывать активации на обратном проходе вместо
            хранения. Примерно -60% памяти за +30% времени. Патчинг сам по себе
            память не экономит (M > N), так что этот флаг -- то, чем она
            возвращается обратно.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.use_checkpoint = use_checkpoint
        self.patching = patch_index is not None

        if self.patching:
            gather_idx = torch.as_tensor(patch_index["gather_idx"], dtype=torch.long)
            unpad_idx = torch.as_tensor(patch_index["unpad_idx"], dtype=torch.long)

            if unpad_idx.numel() != num_nodes:
                raise ValueError(
                    f"индексы патчинга на {unpad_idx.numel()} узлов, "
                    f"модель на {num_nodes}"
                )

            self.num_patches = int(patch_index["num_patches"])
            self.patch_size = int(patch_index["patch_size"])
            if self.num_patches * self.patch_size != gather_idx.numel():
                raise ValueError("R * P не совпадает с числом слотов")

            self.register_buffer("gather_idx", gather_idx)
            self.register_buffer("unpad_idx", unpad_idx)

            # attn_layers_s работает как depth (внутри патча), эти -- как
            # breadth (между патчами). Их пара за слой даёт полную связность
            # уже за два слоя: любой узел достижим по маршруту патч -> позиция.
            self.attn_layers_s_breadth = nn.ModuleList(
                [
                    SelfAttentionLayer(
                        self.model_dim, feed_forward_dim, num_heads, dropout
                    )
                    for _ in range(num_layers)
                ]
            )

    def _apply_layer(self, layer, x, dim):
        """Слой с опциональным пересчётом активаций на обратном проходе."""
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(layer, x, dim, use_reentrant=False)
        return layer(x, dim)

    def _spatial_patched(self, x):
        """Пространственное внимание по патчам вместо плотного по всем узлам.

        Раскладка (batch, in_steps, num_patches, patch_size, model_dim):
          dim=3 -- depth, длина P, ведущая размерность (B, T, R)
          dim=2 -- breadth, длина R, ведущая размерность (B, T, P)

        Слоты-паддинги участвуют во внимании как контекст, но на выходе
        отбрасываются: unpad_idx указывает на слот, которым узел владеет.
        """
        batch_size, in_steps = x.shape[0], x.shape[1]

        x = x.index_select(2, self.gather_idx)  # (batch, steps, num_slots, dim)
        x = x.reshape(
            batch_size, in_steps, self.num_patches, self.patch_size, self.model_dim
        )

        for depth, breadth in zip(self.attn_layers_s, self.attn_layers_s_breadth):
            x = self._apply_layer(depth, x, 3)
            x = self._apply_layer(breadth, x, 2)

        x = x.reshape(batch_size, in_steps, -1, self.model_dim)
        return x.index_select(2, self.unpad_idx)  # (batch, steps, num_nodes, dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = self._apply_layer(attn, x, 1)

        if self.patching:
            x = self._spatial_patched(x)
        else:
            for attn in self.attn_layers_s:
                x = self._apply_layer(attn, x, 2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
