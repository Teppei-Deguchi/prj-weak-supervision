#!/usr/bin/env python3
import argparse
import pandas as pd
import re
from pathlib import Path

# 20 standard amino acids
AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))

# Regular expression for mutation format (e.g., A1L)
MUT_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]$")

def load_mutation_set(csv_path: Path) -> set:
    """Load mutation column from a CSV and return a set of valid mutations"""
    df = pd.read_csv(csv_path)
    if "mutation" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a 'mutation' column")
    muts = (
        df["mutation"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    # Keep only entries that match the expected format
    muts = [m for m in muts if MUT_RE.match(m)]
    return set(muts)

def generate_saturation(wt_seq: str) -> list:
    """Generate all single-site saturation mutations (in L1A format)"""
    wt_seq = wt_seq.upper().strip()
    if not wt_seq or any((aa not in AA20) for aa in wt_seq):
        raise ValueError("WT sequence contains non-standard amino acid characters")
    muts = []
    for i, wt_aa in enumerate(wt_seq, start=1):  # positions are 1-based
        for alt in AA20:
            if alt == wt_aa:
                continue
            muts.append(f"{wt_aa}{i}{alt}")
    return muts

def main():
    ap = argparse.ArgumentParser(
        description="Generate saturation mutagenesis set, "
                    "exclude known mutations, and save to mutation.csv"
    )
    ap.add_argument("--wt-seq", required=True, help="Wild-type sequence (string, required)")
    ap.add_argument("--test",  default=None,       help="CSV with known mutations (default: %(default)s)")
    ap.add_argument("-o", "--output", default="mutation.csv", help="Output CSV file (default: %(default)s)")
    args = ap.parse_args()

    # Get WT sequence
    wt_seq = args.wt_seq.strip().upper()

    # Generate all possible single mutations (saturation)
    all_muts = set(generate_saturation(wt_seq))

    # Load known mutations (train + test)
    known = set()
    
    if args.test is not None:
        test_path  = Path(args.test)
        if test_path.exists():
            known |= load_mutation_set(test_path)
        else:
            print(f"Warning: {test_path} not found, skipping.")

    # Keep only mutations not already known
    remaining = sorted(
        all_muts - known,
        key=lambda x: (int(re.findall(r"\d+", x)[0]), x)
    )

    # Save result
    out_df = pd.DataFrame({"mutation": remaining})
    out_df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  (total {len(remaining)} mutations)")

if __name__ == "__main__":
    main()

