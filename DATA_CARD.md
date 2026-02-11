# APBench data card

## Included file

`data/benchmark/auth_prompt_V1_0_0.json` is the only benchmark data file included in this supplement.

- Records: 2,048
- File size: 2,099,029 bytes
- SHA-256: `06f616734fd45d8d2cc2f523b7d7034e957fa6a89ca08f520ed5d95c0c5266c1`
- Format: UTF-8 JSON array

Distribution summary:

| Field | Counts |
|---|---|
| `user_type` | novice: 1,074; expert: 974 |
| `challenge` | easy: 1,554; medium: 238; hard: 256 |
| `class` | intent_confirmed_all: 1,554; intent_confirmed_prompt_only: 238; intent_not_confirmed: 256 |
| `subject_clear` | true: 1,792; false: 256 |
| source URL domain | Lexica: 1,074; Civitai: 974 |

## Record schema

Each record can contain:

`id`, `index`, `image_url`, `prompt`, `negative_prompt`, `width`, `height`, `llm_score`, `clip_score`, `vqa_score`, `image_score`, `class`, `challenge`, `user_type`, `intent`, `subject_clear`, `sentence_intent`, and `other_metadata`.

`challenge` is the authoritative difficulty annotation (`easy`, `medium`, or `hard`). Evaluation summaries read this field directly and do not infer difficulty from `class`.

`image_url` is provenance metadata pointing to a publicly accessible source page or image endpoint. No image binaries are redistributed. `other_metadata` contains generation settings such as sampler, step count, guidance scale, and seed; it contains no local filesystem paths or account identifiers in this release.

## Privacy and content considerations

The release was scanned for email addresses, personal filesystem paths, account identifiers, and embedded credentials. None were detected. Prompts originate from public text-to-image platforms and may mention public figures, artists, fictional characters, adult themes, or other material present in the original public prompt corpus. Users should review prompts for the requirements of their institution and deployment context.

The stable UUIDs and source URLs are retained for provenance and reproducibility. They are not author or contributor identifiers.

## Licensing and redistribution

Repository code is covered by the top-level `LICENSE`. Source platforms, linked images, model weights, and external benchmarks retain their own terms. This supplement does not grant rights to linked images or third-party model weights. Before public redistribution beyond the paper-review system, confirm that distribution of the benchmark text and metadata is consistent with the source-platform terms and the paper's data-collection protocol.

## Integrity check

```bash
shasum -a 256 data/benchmark/auth_prompt_V1_0_0.json
python -m json.tool data/benchmark/auth_prompt_V1_0_0.json >/dev/null
```
