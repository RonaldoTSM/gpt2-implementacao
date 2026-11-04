# generate.py - gera texto com o nosso GPT.
#
# Com os pesos da OpenAI:
#   python generate.py --pretrained gpt2
# Com um checkpoint do train.py:
#   python generate.py --ckpt ckpt.pt --prompt "ROMEO:"

import argparse

import tiktoken
import torch

from model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--ckpt")
    parser.add_argument("--prompt", default="Hello, I'm a language model,")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.pretrained or args.ckpt, "use --pretrained ou --ckpt"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    if args.pretrained:
        model = GPT.from_pretrained(args.pretrained)
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = GPT(ckpt["config"])
        model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(args.prompt), dtype=torch.long)
    # o mesmo prompt repetido num_samples vezes no batch
    idx = tokens.unsqueeze(0).repeat(args.num_samples, 1).to(device)

    g = torch.Generator(device=device).manual_seed(args.seed)
    out = model.generate(idx, args.max_new_tokens, top_k=50, generator=g)

    for i in range(args.num_samples):
        # o vocabulário do treino tem 50304 posições; as acima de 50257
        # não existem no tokenizador e não dá para decodificar
        ids = [t for t in out[i].tolist() if t < enc.n_vocab]
        print(">", enc.decode(ids))


if __name__ == "__main__":
    main()
