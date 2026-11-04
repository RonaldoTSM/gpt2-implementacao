# train.py - treino do GPT-2 com as técnicas do vídeo:
# acumulação de gradiente, warmup + cosseno, gradient clipping,
# bf16/TF32, torch.compile e DDP.
#
# Treino pequeno (CPU/Mac):
#   python train.py --preset tiny --max_steps 200
# GPT-2 124M numa GPU:
#   python train.py --preset gpt2 --total_batch_size 524288 --B 16 --T 1024 --compile
# Várias GPUs:
#   torchrun --standalone --nproc_per_node=8 train.py --preset gpt2 --total_batch_size 524288 --B 16 --T 1024 --compile

import os
import math
import time
import argparse
import urllib.request

import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tokens(path):
    if not os.path.exists(path):
        print(f"baixando Tiny Shakespeare para {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    return torch.tensor(enc.encode(text))


class DataLoaderLite:
    # Pega B*T + 1 tokens seguidos: x são as entradas e y é o mesmo trecho
    # deslocado em 1 (o próximo token de cada posição).
    # Com DDP, cada processo lê um pedaço diferente.

    def __init__(self, tokens, B, T, process_rank=0, num_processes=1):
        self.tokens = tokens
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # entradas
        y = buf[1:].view(B, T)   # alvos
        self.current_position += B * T * self.num_processes
        # se o próximo batch não cabe, volta ao início
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.reset()
        return x, y


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    # 1) warmup linear
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) depois do fim, fica no mínimo
    if step > max_steps:
        return min_lr
    # 3) no meio, decaimento cosseno de max_lr até min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # vai de 1 a 0
    return min_lr + coeff * (max_lr - min_lr)


PRESETS = {
    # modelo pequeno para rodar no notebook
    "tiny": GPTConfig(block_size=128, vocab_size=50304, n_layer=4, n_head=4, n_embd=128),
    # GPT-2 124M. 50304 em vez de 50257 porque é múltiplo de 128 (mais rápido na GPU)
    "gpt2": GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS.keys(), default="tiny")
    parser.add_argument("--data", default="input.txt")
    parser.add_argument("--B", type=int, default=8)   # micro-batch
    parser.add_argument("--T", type=int, default=128)  # tamanho da sequência
    parser.add_argument("--total_batch_size", type=int, default=8 * 128 * 4)  # tokens por passo (GPT-2: 524288)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", default="ckpt.pt")
    parser.add_argument("--log", default="log.txt")
    args = parser.parse_args()

    # DDP: o torchrun define RANK, LOCAL_RANK e WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    master_process = ddp_rank == 0  # só ele imprime e salva
    device_type = "cuda" if device.startswith("cuda") else device
    if master_process:
        print(f"dispositivo: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # acumulação de gradiente: vários micro-batches antes de cada optimizer.step()
    B, T = args.B, args.T
    assert args.total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"batch total: {args.total_batch_size} | grad_accum_steps: {grad_accum_steps}")

    tokens = load_tokens(args.data)
    n = int(0.9 * len(tokens))  # 90% treino, 10% validação
    train_loader = DataLoaderLite(tokens[:n], B, T, ddp_rank, ddp_world_size)
    val_loader = DataLoaderLite(tokens[n:], B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"{len(tokens):,} tokens ({n:,} treino / {len(tokens) - n:,} validação)")

    torch.set_float32_matmul_precision("high")  # TF32

    config = PRESETS[args.preset]
    model = GPT(config).to(device)
    raw_model = model  # sem compile/DDP, para salvar e configurar o otimizador
    if master_process:
        print(f"parâmetros: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.max_lr,
                                               device_type=device_type)
    min_lr = args.max_lr * 0.1
    use_bf16 = device_type == "cuda"  # bf16 só em GPU

    if master_process:
        open(args.log, "w").close()  # zera o log

    for step in range(args.max_steps):
        t0 = time.time()
        last_step = step == args.max_steps - 1

        # avaliação na validação de tempos em tempos
        if step % args.eval_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach() / val_loss_steps
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step:5d} | val loss {val_loss_accum.item():.4f}")
                with open(args.log, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            model.train()

        # passo de treino
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # só sincroniza os gradientes entre GPUs no último micro-passo
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, loss = model(x, y)
            # o backward soma os gradientes; dividir faz virar média
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # gradient clipping na norma global
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, args.max_lr, min_lr, args.warmup_steps, args.max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        # espera a GPU terminar para medir o tempo certo
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt
        if master_process:
            print(f"step {step:5d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                  f"norm {norm:.3f} | {dt * 1000:.0f}ms | {tokens_per_sec:,.0f} tok/s")
            with open(args.log, "a") as f:
                f.write(f"{step} train {loss_accum.item():.4f}\n")

    if master_process:
        torch.save({"model": raw_model.state_dict(), "config": raw_model.config}, args.out)
        print(f"checkpoint salvo em {args.out}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
