# model.py - arquitetura do GPT-2 (124M), seguindo o vídeo
# "Let's reproduce GPT-2 (124M)" do Andrej Karpathy.
#
# Fluxo: tokens -> wte + wpe -> 12 blocos (atenção + MLP) -> ln_f -> lm_head -> logits

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # tamanho máximo do contexto
    vocab_size: int = 50257  # 50.000 merges BPE + 256 bytes + <|endoftext|>
    n_layer: int = 12        # número de blocos
    n_head: int = 12         # cabeças de atenção
    n_embd: int = 768        # dimensão do embedding


class CausalSelfAttention(nn.Module):
    # Atenção multi-cabeça com máscara causal: cada token olha só para
    # ele mesmo e para os anteriores.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K e V calculados numa única Linear (C -> 3C)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # projeção de saída, de volta para o fluxo residual
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # usado no _init_weights
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch, sequência, embedding
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # separa as cabeças: (B, T, C) -> (B, nh, T, hs)
        # assim cada cabeça calcula sua atenção em paralelo
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Versão manual (o que a função abaixo calcula):
        #   att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #   att = att.masked_fill(mascara[:, :, :T, :T] == 0, float('-inf'))
        #   att = F.softmax(att, dim=-1)
        #   y = att @ v
        # Dividir por sqrt(hs) evita que o softmax sature.
        # O -inf da máscara vira peso 0 depois do softmax (não vê o futuro).
        # Em GPU, scaled_dot_product_attention usa FlashAttention.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # junta as cabeças de novo: (B, nh, T, hs) -> (B, T, C)
        # contiguous() porque o transpose não reorganiza a memória e o view exige isso
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    # Aplicado a cada token separadamente: expande para 4C, GELU, volta para C.

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # aproximação tanh: a mesma usada pela OpenAI no GPT-2
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # Bloco pré-norm: a LayerNorm vem antes da atenção e do MLP, e o
    # caminho residual (x + ...) fica limpo de ponta a ponta, o que ajuda
    # o gradiente a chegar nas primeiras camadas.

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # tokens trocam informação
        x = x + self.mlp(self.ln_2(x))   # cada token processa sozinho
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # nomes iguais aos do checkpoint do HuggingFace, para o from_pretrained
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # embedding de token
            wpe=nn.Embedding(config.block_size, config.n_embd),  # embedding de posição
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: entrada e saída usam a mesma matriz (economiza ~38M parâmetros)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # cada bloco soma 2 vezes no residual (attn e mlp), então a variância
            # cresce com a profundidade; escalamos por 1/sqrt(2 * n_layer)
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T) com os índices dos tokens
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequência {T} maior que block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, C)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        x = tok_emb + pos_emb                # broadcast na dimensão B
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # cross_entropy espera (N, V) e (N,), por isso o view
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        # carrega os pesos do GPT-2 da OpenAI (via HuggingFace) no nosso modelo
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("carregando pesos pré-treinados de %s" % model_type)

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = list(sd.keys())

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # .attn.bias e .attn.masked_bias são a máscara do HF, não são pesos
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]
        # a OpenAI usou Conv1D, que guarda esses pesos transpostos em relação ao nn.Linear
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys_hf) == len(sd_keys), f"chaves diferentes: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # weight decay só nos tensores 2D (matrizes); biases e LayerNorm ficam sem
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        print(f"tensores com decay: {len(decay_params)} ({sum(p.numel() for p in decay_params):,} parâmetros)")
        print(f"tensores sem decay: {len(nodecay_params)} ({sum(p.numel() for p in nodecay_params):,} parâmetros)")
        # versão fused (um kernel só) quando existe e estamos em CUDA
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # betas e eps do paper do GPT-3
        return torch.optim.AdamW(optim_groups, lr=learning_rate,
                                 betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50, generator=None):
        # gera um token por vez e anexa à sequência
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # corta no contexto máximo
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # só a última posição importa
            probs = F.softmax(logits, dim=-1)
            # top-k 50 (como o pipeline do HF): sorteia só entre os 50 mais prováveis
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=generator)
            next_token = torch.gather(topk_indices, -1, ix)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
