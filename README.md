# Semantically-Diverse-Deterministic-Sampling

## 📌 Overview
Traditional sampling methods for large language models (LLMs) like GPT-4 often rely on naive approaches such as temperature or top-k sampling. While simple, these techniques can be computationally expensive and frequently produce outputs that are either too similar (lack diversity) or too unlikely (poor quality).

This project implements a smart sampling method that combines:

- Deterministic decoding (temperature=0) for the highest-probability base generation.

- Targeted semantic divergence: identifying alternative tokens that are both sufficiently different and still reasonably probable.

- Threshold-driven selection, enabling fine control over the tradeoff between diversity and likelihood.

This approach is particularly suited for scenarios like pass@k evaluation in code generation, creative writing, or any application requiring multiple plausible, meaningfully different outputs.

## ⚙️ How It Works
1- Initial greedy decode:

- Generate a base output using temperature=0, ensuring the highest-probability token path.

2- Identify alternative tokens:

- For each decoding step, look at tokens not selected by the greedy decode.

- Filter tokens by:

  - Probability threshold: Must have a likelihood above a defined cutoff.

  - Semantic divergence threshold: Must differ significantly in meaning from the selected greedy token, measured via cosine distance on token or contextual embeddings.

3- Generate alternative completions:

- For each accepted divergent token, continue decoding deterministically from that point to produce a new, coherent completion.

4- Collect final candidates:

- Produce a diverse set of high-quality outputs suitable for downstream evaluation or pass@k metrics.

## ✨ Key Features
✅ More meaningful diversity:
Avoid near-duplicates by explicitly enforcing semantic distance.

✅ High likelihood:
By only considering reasonably probable alternative tokens, maintain overall quality.

✅ Customizable thresholds:
Control how adventurous vs conservative the sampling is.

✅ Pluggable divergence metric:
Use simple embedding cosine similarity, contextual embeddings, or advanced sentence encoders.
