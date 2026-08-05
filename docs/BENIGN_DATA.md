# Datasets: wiring paper-backed logs into the verdict task

The verdict task needs **both** classes — `attack` and `benign` — per domain
(endpoint / network / cloud). We get them from **public, paper-backed datasets**
converted into one normalized format. No Splunk Attack Range required.

> We evaluated Splunk Attack Range for matched benign/malicious capture and
> deliberately dropped it: it's a cloud tool with a cloud bill and a long Docker
> tail, and paper-backed datasets give matched, citable, reproducible data for
> free. (The war story lives in the blog post, not the pipeline.)

---

## The one format everything becomes
Every dataset is converted **once** into a normalized `.jsonl`, one record/line:
```json
{"log": "...", "technique": "T1110", "sourcetype": "auth.log", "domain": "endpoint", "is_malicious": true}
```
A single file can hold **both** classes (each record has its own `is_malicious`).
The pipeline reads these via `FEDDAPT_LOG_SOURCES`; `build_clients` splits them by
class into `client_*_logs.json` (malicious) and `client_*_benign.json` (benign).

Convert with `scripts/convert_dataset.py`, then point the env var at the output:
```bash
python scripts/convert_dataset.py --dataset ait --src /data/AIT/mail.cup.com --out data/normalized/ait_mail.jsonl
# in .env:
FEDDAPT_LOG_SOURCES=data/normalized        # a directory picks up every *.jsonl
python scripts/build_data.py --teacher claude-haiku-4-5
```
Converters **fail loudly** if they parse zero usable records — a silent-empty
source is a bug, not an empty dataset (we've been burned by exactly that).

---

## Recommended sources per domain

### Endpoint + network — AIT Log Data Set  ✅ wired
Landauer et al., *"Maintainable Log Datasets for Evaluation of IDS,"* IEEE TDSC
2022. Zenodo `10.5281/zenodo.5789064` (V2). CC-BY. An enterprise testbed with
state-machine-simulated **normal user behavior** and **injected multi-step
attacks**, labeled per log line. Raw logs (auth, audit, apache, dns, suricata…),
so both classes come from **one matched environment** — the leak-proof property.

```bash
python scripts/convert_dataset.py --dataset ait \
    --src /data/AIT-LDS-v2/<one-dataset-dir> --out data/normalized/ait_<name>.jsonl
```
`--src` is a single dataset dir (the one containing `gather/` and `labels/`). We
window logs into records and mark a window `attack` iff it contains ≥1 labeled
attack line — i.e. *"does this activity contain evidence of an attack?"*, the
verdict question. Tune `--window`, `--max-benign-per-file`, `--max-malicious-per-file`.

### Network — CIC-IDS2017 / UNSW-NB15  ✅ wired
Sharafaldin et al., ICISSP 2018 / Moustafa & Slay, MilCIS 2015. Labeled flow CSVs
(benign + attack from one testbed). The `log` becomes a compact `k=v` flow record.
```bash
python scripts/convert_dataset.py --dataset cic --src /data/CIC-IDS2017/csv --out data/normalized/cic.jsonl
```
Note: these are tabular flow features, not raw text — fine as a network baseline,
but AIT is the more log-native option if you want one source for endpoint+network.

### Cloud — CloudTrail (Stratus Red Team detonations)  ✅ wired
One class per export: `--label attack` for a Stratus detonation log, `--label
benign` for a normal-account CloudTrail export.
```bash
python scripts/convert_dataset.py --dataset cloudtrail --label attack --src /data/stratus --out data/normalized/cloud_attack.jsonl
python scripts/convert_dataset.py --dataset cloudtrail --label benign --src /data/acct-ct --out data/normalized/cloud_benign.jsonl
```

### Must-cite, likely request-access — Multi-Source Cybersecurity Logs
Niloy et al., arXiv **2606.18190** (Jun 2026). Windows endpoint + network + browser,
870 sessions (70 attack / 800 benign), ATT&CK-labeled, and they LoRA-fine-tune SLMs
for chunk-classification + technique-ID — essentially our `verdict` + `explain_log`.
No public download link in the paper, so treat it as **mandatory Related Work**
(and near-twin prior art) rather than a wire-in. If they release data, it drops in
via a small converter.

---

## The matched-environment rule (still applies)
Balance is **per domain**: within each domain you evaluate, benign and malicious
must come from the *same* environment, or the model learns to fingerprint
environments instead of attacks. Cross-domain differences are expected. Each
recommended source above is internally matched (one testbed → both classes), which
is exactly why they're safe. Don't pair a benign dataset from environment A with a
malicious dataset from environment B within the same domain.

OS (Windows/Linux/macOS) is **not** a domain here — `_domain_of` collapses all of
them into `endpoint`. So OS diversity is optional realism, not a balance
requirement; macOS in particular has no good matched attack data — skip it.
