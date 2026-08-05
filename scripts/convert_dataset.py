"""Convert a public, paper-backed log dataset into a normalized .jsonl source.

Run this ONCE per dataset. Point FEDDAPT_LOG_SOURCES at the resulting .jsonl
files (comma-separated, or a directory of them), then run build_data.py as usual.

    # AIT (endpoint + network, both classes; Zenodo 10.5281/zenodo.5789064)
    python scripts/convert_dataset.py --dataset ait \
        --src /data/AIT-LDS-v2/mail.cup.com --out data/normalized/ait_mail.jsonl

    # CIC-IDS2017 / UNSW-NB15 (network flows, both classes)
    python scripts/convert_dataset.py --dataset cic \
        --src /data/CIC-IDS2017/csv --out data/normalized/cic.jsonl

    # CloudTrail (cloud; one class per export — run twice)
    python scripts/convert_dataset.py --dataset cloudtrail --label attack \
        --src /data/stratus-detonations --out data/normalized/cloud_attack.jsonl
    python scripts/convert_dataset.py --dataset cloudtrail --label benign \
        --src /data/my-account-ct     --out data/normalized/cloud_benign.jsonl

Then:  FEDDAPT_LOG_SOURCES=data/normalized   (a dir picks up every *.jsonl)
"""
import argparse

from fedapt import datasets as D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(D.CONVERTERS),
                    help="which dataset converter to run")
    ap.add_argument("--src", required=True, help="path to the downloaded dataset")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    # dataset-specific (only the relevant ones are used)
    ap.add_argument("--label", choices=["attack", "benign"],
                    help="cloudtrail only: class of this export")
    ap.add_argument("--window", type=int, default=20, help="ait: lines per record")
    ap.add_argument("--max-benign-per-file", type=int, default=200, help="ait")
    ap.add_argument("--max-malicious-per-file", type=int, default=400, help="ait")
    ap.add_argument("--max-per-class", type=int, default=4000, help="cic")
    args = ap.parse_args()

    if args.dataset == "ait":
        recs = D.convert_ait(args.src, window=args.window,
                             max_benign_per_file=args.max_benign_per_file,
                             max_malicious_per_file=args.max_malicious_per_file)
    elif args.dataset == "cic":
        recs = D.convert_cic(args.src, max_per_class=args.max_per_class)
    elif args.dataset == "cloudtrail":
        if not args.label:
            raise SystemExit("--label attack|benign is required for cloudtrail")
        recs = D.convert_cloudtrail(args.src, label=args.label)
    else:                                                  # pragma: no cover
        raise SystemExit(f"unknown dataset {args.dataset}")

    D.write_jsonl(recs, args.out)


if __name__ == "__main__":
    main()
