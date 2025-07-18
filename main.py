import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

MODEL_NAME = "google/gemma-2b-it"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
PROB_THRESHOLD = 0.01
SEMANTIC_DISTANCE_THRESHOLD = 0.3
MAX_NEW_TOKENS = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.to(device)
model.eval()

embedder = SentenceTransformer(EMBED_MODEL)

def decode_tokens(input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def get_cosine_divergence(base_str, alt_str):
    base_vec = embedder.encode(base_str, convert_to_tensor=True)
    alt_vec = embedder.encode(alt_str, convert_to_tensor=True)
    cosine_sim = util.cos_sim(base_vec, alt_vec).item()
    return 1.0 - cosine_sim

def greedy_generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    return output[0]

def get_divergent_tokens(prompt, base_ids, k):
    alt_completions = []
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cur_input = input_ids.clone()

    for i in range(len(base_ids) - input_ids.shape[1]):
        if i == k:
            break
        with torch.no_grad():
            outputs = model(cur_input)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1).squeeze()

        base_token = base_ids[input_ids.shape[1] + i]
        topk_probs, topk_ids = torch.topk(probs, k=TOP_K)

        for prob, tok_id in zip(topk_probs, topk_ids):
            if tok_id == base_token:
                continue
            if prob.item() < PROB_THRESHOLD:
                continue

            alt_ids = torch.cat([cur_input.squeeze(), tok_id.unsqueeze(0)], dim=0).unsqueeze(0)

            with torch.no_grad():
                alt_out = model.generate(
                    alt_ids,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id
                )

            alt_text = decode_tokens(alt_out[0])
            base_text = decode_tokens(base_ids)

            divergence = get_cosine_divergence(base_text, alt_text)
            if divergence >= SEMANTIC_DISTANCE_THRESHOLD:
                alt_completions.append((alt_text, divergence, prob.item()))
        
        next_token = base_ids[input_ids.shape[1] + i].unsqueeze(0).unsqueeze(0)
        cur_input = torch.cat([cur_input, next_token], dim=1)

    return alt_completions

def run_smart_sampling(prompt):
    print("🔹 Base Greedy Completion:")
    base_ids = greedy_generate(prompt)
    base_text = decode_tokens(base_ids)
    print(base_text)

    print("\n🔹 Searching for Semantically Divergent Alternatives...")
    alts = get_divergent_tokens(prompt, base_ids, 3)

    print(f"\n🔹 Found {len(alts)} alternatives passing thresholds.\n")
    for i, (text, div, prob) in enumerate(alts):
        print(f"[Alt #{i+1}] (Prob: {prob:.3f}, Divergence: {div:.3f})")
        print(text)
        print("-" * 60)

if __name__ == "__main__":
    user_prompt = "Once upon a time in a quiet village,"
    run_smart_sampling(user_prompt)
