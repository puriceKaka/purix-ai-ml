# PURIX AI Model

PURIX is a from-scratch, CPU-compatible decoder-only Transformer language-model experiment. It has a legacy word-and-punctuation tokenizer and an optional versioned byte-level BPE tokenizer, alongside its PyTorch model, training loop, checkpoint format, and autoregressive sampler. It does not use pretrained model weights, hosted AI APIs, or execute commands.

The model contains token and learned position embeddings, causal multi-head self-attention, feed-forward blocks, residual connections, normalization, and a next-token output head. The causal attention mask prevents access to future tokens. The legacy word tokenizer lowercases text and separates punctuation; the optional byte-level BPE tokenizer preserves case and whitespace. Both use stable PAD, UNK, BOS, and EOS IDs.

## Preserve the original checkpoint

`checkpoints/purix.pt` is the pre-existing model and remains unchanged. Word-tokenizer checkpoints use a matching `.vocab.json`; byte-level BPE checkpoints use `.tokenizer.json`. Checkpoints record tokenizer type and format version, so loading selects the correct implementation. Historical checkpoints without tokenizer metadata continue to load as word-tokenizer checkpoints. “Best” is selected by document-level validation loss and “latest” is the most recent evaluation step. Checkpoints record SHA-256 hashes for training text, held-out text, and tokenizer state. Resume rejects changed data/configuration. Evaluation verifies the checkpoint hashes against both supplied corpus files before reporting held-out loss.

## Baseline data

`data/baseline/v1` is a reproducibly extracted, source-separated next-token baseline. Training is the official NIST SP 800-53 Release 5.2.0 machine-readable control catalog (261,128 whitespace tokens); validation is the separate NIST SP 800-61 Rev. 3 incident-response publication (13,843 whitespace tokens). The source URLs, publication information, SHA-256 hashes, extraction method, text hashes, and tokenizer/OOV diagnostics are in [manifest.json](data/baseline/v1/manifest.json). Training and validation are separate publications; they are not random chunks of one document. The baseline tokenizer sees training text only. Validation UNK rate is recorded because this word-level tokenizer has limited coverage. These sources were acquired from NIST; the application reads local files only.

The included sources support a reproducible language-model baseline, not a claim that two technical publications represent general English, a full cyber corpus, or an independently curated capability benchmark. Check source versions and license/usage terms before adding other documents. Do not place private records or credentials in the corpus. The trusted documents under `knowledge/documents` are kept locally for retrieval; no prompts or documents are uploaded.

## PowerShell commands

Run from the project root with the existing virtual environment:

```powershell
# Verify the implementation
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Inspect local files in data/raw without writing output
.\.venv\Scripts\python.exe -m src.prepare_data --input data/raw --output-root data/experiments/v1 --target-mb 5 --max-clean-mb 10 --dry-run

# Put at least two distinct, sufficiently long source documents in data/raw first.
# sources.json records metadata for those files; it does not contain their text.
# Copy and edit provenance metadata for the documents you place in data/raw
Copy-Item data/raw/sources.example.json data/raw/sources.json

# Prepare a reproducible local corpus (no network access). Stop if this fails;
# do not run training unless both output files exist.
.\.venv\Scripts\python.exe -m src.prepare_data --input data/raw --output-root data/experiments/v1 --sources data/raw/sources.json --target-mb 5 --max-clean-mb 10 --validation-fraction 0.05 --seed 42
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed; training was not started." }
if (-not (Test-Path data/experiments/v1/datasets/train/training.txt) -or -not (Test-Path data/experiments/v1/datasets/validation/validation.txt)) { throw "Training and validation corpora are missing; training was not started." }

# Resume after interruption; unchanged per-document cleaned files are verified and reused
.\.venv\Scripts\python.exe -m src.prepare_data --input data/raw --output-root data/experiments/v1 --sources data/raw/sources.json --target-mb 5 --max-clean-mb 10 --validation-fraction 0.05 --seed 42 --resume

# Optional network collection: copy/edit a license-reviewed manifest; dry-run makes no request
Copy-Item data/downloads.example.json data/downloads.json
.\.venv\Scripts\python.exe -m src.prepare_data --download-manifest data/downloads.json --max-download-mb 5 --output-root data/experiments/v1 --target-mb 5 --max-clean-mb 10 --dry-run

# Re-extract the documented baseline after changing its source files
.\.venv\Scripts\python.exe -m src.build_baseline

# Train an isolated short CPU smoke run; writes only beneath experiments/smoke_verified
.\.venv\Scripts\python.exe -m src.train --corpus data/training.txt --validation-corpus data/validation.txt --checkpoint experiments/smoke_verified/purix_latest.pt --best-checkpoint experiments/smoke_verified/purix_best.pt --steps 5 --batch-size 2 --block-size 16 --d-model 32 --heads 2 --layers 1 --dropout 0 --seed 42

# Train a new corpus experiment after reviewing its report; this does not write purix.pt
.\.venv\Scripts\python.exe -m src.train --corpus data/experiments/v1/datasets/train/training.txt --validation-corpus data/experiments/v1/datasets/validation/validation.txt --checkpoint experiments/v1/purix_latest.pt --best-checkpoint experiments/v1/purix_best.pt --steps 100 --batch-size 4 --block-size 32 --d-model 64 --heads 4 --layers 2 --seed 42

# Optional BPE run: fits tokenizer only from --corpus and writes a separate experiment
.\.venv\Scripts\python.exe -m src.train --corpus data/experiments/general-v1/datasets/train/training.txt --validation-corpus data/experiments/general-v1/datasets/validation/validation.txt --checkpoint experiments/general-bpe-v1/purix_latest.pt --best-checkpoint experiments/general-bpe-v1/purix_best.pt --tokenizer-type bytelevel_bpe --tokenizer-vocab-size 8192 --tokenizer-min-frequency 2 --steps 100 --batch-size 4 --block-size 64 --d-model 64 --heads 4 --layers 2 --seed 42

# Stage 1 general-language preparation, after source approval and local collection.
# The preparer records source license, attribution, document ID, hashes, and a
# document-level split. It does not download anything in this form.
.\.venv\Scripts\python.exe -m src.prepare_data --input data/general-v1/raw --sources data/general-v1/sources.json --output-root experiments/general-v1/dataset --target-mb 5 --max-clean-mb 10 --validation-fraction 0.1 --seed 20261020

# Later: isolated CPU smoke training. This refuses to overwrite existing files;
# use --resume only with the matching latest checkpoint and unchanged corpus.
.\.venv\Scripts\python.exe -m src.train --corpus experiments/general-v1/dataset/datasets/train/training.txt --validation-corpus experiments/general-v1/dataset/datasets/validation/validation.txt --checkpoint experiments/general-v1/purix_latest.pt --best-checkpoint experiments/general-v1/purix_best.pt --tokenizer-type bytelevel_bpe --tokenizer-vocab-size 8000 --tokenizer-min-frequency 2 --steps 20 --batch-size 2 --block-size 64 --d-model 64 --heads 4 --layers 2 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.01 --validation-interval 5 --checkpoint-interval 5 --device cpu --seed 20261020

# Later: held-out loss plus raw, seeded responses for the fixed general suite.
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint experiments/general-v1/purix_best.pt --training-corpus experiments/general-v1/dataset/datasets/train/training.txt --validation-corpus experiments/general-v1/dataset/datasets/validation/validation.txt --prompts-file data/evaluation/general_v1.jsonl --seed 1234 --max-new-tokens 40 --temperature 0.8 --top-k 20 --output experiments/general-v1/evaluation_smoke.json

# Score only after that command has created experiments/v1/purix_best.pt
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint experiments/v1/purix_best.pt --training-corpus data/experiments/v1/datasets/train/training.txt --validation-corpus data/experiments/v1/datasets/validation/validation.txt --seed 1234 --prompt "A security control should" --prompt "Incident response begins"

# Evaluate fixed assistant checks with local retrieval; outputs include retrieved source metadata
.\.venv\Scripts\python.exe -m src.evaluate_assistant --baseline experiments/v1/purix_best.pt --cases data/evaluation/assistant.jsonl --knowledge-dir knowledge/documents --seed 1234

# Optional instruction tuning. This writes its own latest/best files and preserves the base checkpoint.
.\.venv\Scripts\python.exe -m src.instruction_tune --base-checkpoint experiments/v1/purix_best.pt --train data/instructions/train.jsonl --validation data/instructions/validation.jsonl --latest experiments/v1/purix_instruction_latest.pt --best experiments/v1/purix_instruction_best.pt --steps 100

# Compare base and instruction-tuned behavior on the same fixed cases
.\.venv\Scripts\python.exe -m src.evaluate_assistant --baseline experiments/v1/purix_best.pt --instruction experiments/v1/purix_instruction_best.pt --cases data/evaluation/assistant.jsonl --knowledge-dir knowledge/documents --seed 1234

# Interactive local chat with retrieved citations; use /exit to leave
.\.venv\Scripts\python.exe -m src.chat --checkpoint experiments/v1/purix_best.pt

# Reproduce the four fixed chat diagnostics. JSON keeps raw model text separate
# from a retrieved-evidence fallback or an unavailable result.
.\.venv\Scripts\python.exe -m src.chat --checkpoint checkpoints/purix.pt --diagnostic --max-new-tokens 40 --temperature 0.7 --top-k 20 --repetition-penalty 1.2 --no-repeat-ngram-size 3
```

Training defaults are small for CPU (width 128, 4 heads, 4 layers, context 64, batch 16). Larger width/layer counts increase compute and memory; raising context length increases attention cost quadratically. Do not increase steps to chase a lower training loss without inspecting held-out metrics. Training logs distinguish a sampled training minibatch loss from document-level held-out loss.

### Corpus preparation

Put original local files in `data/raw/`. Supported inputs are UTF-8 `.txt`, `.md`/`.markdown`, HTML, PDF, and DOCX. PDF extraction uses the already-installed `pypdf`; DOCX and HTML use Python's standard library. The preparer writes an isolated experiment tree with `raw/` source copies, `cleaned/` deduplicated per-document text, `datasets/train/training.txt`, `datasets/validation/validation.txt`, `sources.json`, `prepare_state.json`, and `reports/dataset_report.json`. The source files outside this tree are not changed. Per-document writes are atomic and `--resume` checks the run configuration and verifies existing cleaned-file hashes.

The default cleaned text target is 5 MiB and hard cap is 10 MiB. Documents are deduplicated by normalized cleaned content and assigned to train/validation by document hash before corpus files (and later next-token windows) are generated. The report includes raw/cleaned bytes, estimated tokenizer token counts, exact word/punctuation vocabulary size from training documents, excluded/duplicate files, provenance fields, and a train/validation overlap check. At least two distinct eligible documents are required for a real split. Dry-run performs extraction and reporting without writing output; with an empty input directory it reports zero eligible documents.

I ran the pipeline on the NIST PDFs already present in `knowledge/documents` (no network download): [dataset report](data/experiments/nist-defensive-v1/reports/dataset_report.json). It produced 1,848,589 cleaned bytes from three documents, with two training documents, one held-out document, 7,528 training-vocabulary entries, and zero exact overlap. Every document is defensive cybersecurity material; the held-out SP 800-92 source is from 2006. This run validates ingestion and document splitting, but it is **not** the requested general-English baseline. The available workspace does not contain a licensed general-English corpus, so none was fabricated or downloaded.

Optional downloads require a user-provided JSON manifest. Accepted identifiers are `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, and `Public-Domain-US`; entries must also include a license URL and `intended_use_permitted: true`. The preparer records this declaration and retrieval date but does not independently verify legal rights. The download cap is separate from the cleaned-corpus cap and is hard-limited to 10 MiB; URLs must be HTTPS and end in `.txt`, `.md`, or `.markdown`, and size is checked while streaming. No download is started by default. `data/raw/sources.example.json` shows local provenance fields.

For NIST material, NIST describes technical-series reuse terms and notes that third-party material can have separate rights; review document notices and applicable local requirements before reuse. See [NIST technical-series copyright and licensing](https://www.nist.gov/research-library/nist-publications).

The existing tokenizer is **word-and-punctuation based**. It lowercases, splits punctuation into separate tokens, and maps every unseen token to the single `<UNK>` ID (1). A separate optional byte-level BPE tokenizer is available for new experiments. Install its dependency with `python -m pip install -r requirements-tokenizer.txt`, then fit it only from an explicitly supplied training file with `BPETokenizer.train_from_file(training_corpus, vocab_size=8192)`. It preserves case and spacing and represents unseen UTF-8 text as byte-level pieces. Its vocabulary changes the token embedding and tied output-head dimensions, so it cannot be applied to old weights unchanged. Train any model using it under a new `experiments/<name>/` directory; do not replace `checkpoints/purix.pt` or its tokenizer. The BPE tokenizer rejects paths marked as validation, evaluation, test, or held-out data and stores the training corpus hash in its tokenizer metadata.

`src.train` loads token IDs and overlapping-window views into memory. On CPU, the output logits scale approximately with `batch_size × block_size × vocabulary_size` (float32), while model parameters and AdamW state scale with vocabulary and width. The training log now reports observed step/token-position throughput, elapsed time, losses, and peak process RSS where Windows exposes it. The 5-step smoke command above writes only under `experiments/smoke`; it is a telemetry check on the existing small corpus, not a general-language baseline or evidence of learning quality. The 1–10 MiB dataset should only be trained after its source report and license metadata have been reviewed.

## Instruction tuning and retrieval

Instruction examples are JSONL records in `data/instructions`. With a word tokenizer, the trainer appends new training vocabulary IDs and initializes only those added embedding rows; with BPE, it reuses the base tokenizer. In both cases prompt labels are masked with `-100`, so cross-entropy is computed on the response and EOS only. Validation examples are separate. The instruction model is stored separately. `data/evaluation/assistant.jsonl` is not used for training and gives fixed prompts, expected terms, retrieval queries, and a small obvious-harm request set.

`src.knowledge` indexes local PDF, JSON/OSCAL, Markdown, and text files into overlapping passages and ranks them with a local BM25-like keyword score. Index metadata is cached in `knowledge/.purix_knowledge_index.json` and invalidated when source files change. `src.chat` puts short retrieved excerpts in the model prompt and prints source titles, pages, URLs, and excerpts separately. This is retrieval assistance, not guaranteed grounding: the tiny model may ignore context or hallucinate. The refusal gate blocks a small set of clearly harmful requests; it is not a comprehensive security policy. No shell, exploitation, or external-system actions are implemented.

Retrieval removes exact chunks during indexing and suppresses near-duplicate overlapping results during ranking, while retaining distinct passages from the same document. A bare numeric match cannot retrieve a passage, and queries with more than two meaningful terms require at least two term matches. Chat also applies conservative, tested rejection checks for empty, excessively repetitive, punctuation-heavy, or off-topic completions. These checks do not establish factual correctness. When they reject a completion, PURIX shows a concise local excerpt prefixed `Retrieved evidence (not model-generated):`; if there is no relevant local evidence, it says that it could not produce a usable answer. The fixed diagnostic command records both the raw completion and the displayed result.

## Evaluation limits

The tests cover data/hash checks, checkpoint/tokenizer loading, deterministic scoring and sampling, causal masking, response-only label masks, local retrieval ranking, context insertion, and basic refusal behavior. They establish code behavior only. Next-token loss, keyword recall, retrieval hit rate, coherent text, and a handful of prompt checks do not establish factual accuracy, instruction-following quality, safety, or cybersecurity capability. PURIX remains experimental and is not represented as capable of cybersecurity analysis.

## Local agent foundation (Stage 2)

`src.agent` provides an injectable `AgentOrchestrator`, a conservative keyword router, typed local events with an append-only SQLite store, and a deterministic simulated monitoring provider. `create_model_orchestrator` in `src.agent.handlers` now connects an already-loaded model and tokenizer to separate conversation and defensive-analysis handlers. The CLI still loads checkpoints through `src.chat.load_checkpoint`; handlers reuse `src.chat.answer_question` for generation, refusal checks, and local retrieval. Defensive responses carry actual retrieved source metadata and mark `tool_action_occurred` false. No handler executes scans or inspects live systems.

Agent callers should provide an explicit `conversation_id` to `AgentOrchestrator.handle` to retain bounded follow-up context. Calls without an ID are isolated and do not retain history. `clear_conversation(id)` drops a conversation's in-memory context. Conversation state is not stored in SQLite. Simulated monitoring remains separate from chat. Event storage defaults to `:memory:`; a countdown completion is explicitly simulated and records `system_action_taken: false`, and no operating-system command is run. The router is a simple keyword baseline, not a trained classifier. This remains a Python API; there is no desktop frontend, voice output, real monitoring, or security tool integration. Run its tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_foundation -v
```
