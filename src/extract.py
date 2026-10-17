"""Extract hidden-state activations for every prompt.

Usage:
  python -m src.extract --model Qwen/Qwen2.5-7B --out results/acts_qwen7b.npz
  python -m src.extract --model tiny-random --out results/acts_tiny.npz   # smoke test, no download

Saves, per layer, the last-token and mean-pooled residual stream for each prompt,
plus model id, revision, dtype, device and prompt order for provenance.
"""
import argparse, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def tiny_random(seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    torch.manual_seed(seed)
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {"[UNK]": 0, "[PAD]": 1}
    return vocab, tok

def load(model_id, device, dtype):
    if model_id == "tiny-random":
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        torch.manual_seed(0)
        rows = [json.loads(l) for l in open("data/prompts_v3.jsonl", encoding="utf-8")]
        t = Tokenizer(models.WordLevel(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace()
        t.train_from_iterator([r["text"] for r in rows], trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
        tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]")
        cfg = Qwen2Config(vocab_size=t.get_vocab_size(), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        return tok, Qwen2ForCausalLM(cfg).eval(), "tiny-random(seed=0)"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return tok, m, getattr(m.config, "_name_or_path", model_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", default="data/prompts_v3.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit: rows = rows[: a.limit]
    tok, model, name = load(a.model, device, dtype)
    tok.padding_side = "right"
    last, mean = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            texts = [r["text"] for r in rows[i:i + a.batch]]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states  # (L+1) x B x T x D
            H = torch.stack(hs[1:], 1).float()  # B x L x T x D
            mask = enc["attention_mask"].float()
            idx = mask.sum(1).long() - 1
            last.append(H[torch.arange(H.shape[0]), :, idx].cpu().numpy())
            mean.append(((H * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]).cpu().numpy())
            print(f"{min(i + a.batch, len(rows))}/{len(rows)}", flush=True)
    np.savez_compressed(a.out, last=np.concatenate(last), mean=np.concatenate(mean),
                        ids=np.array([f"{r['scenario_id']}|{r['template']}|{r['frame']}" for r in rows]),
                        meta=json.dumps({"model": name, "device": device, "dtype": str(dtype), "n": len(rows),
                                         "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}))
    print("saved", a.out)

if __name__ == "__main__":
    main()
