import argparse
import pandas as pd
import re
from pathlib import Path
from typing import Optional

MUT_PAT = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$')

def read_first_fasta_seq(path: Path) -> str:
    seq = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            seq.append(line)
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return ''.join(seq).strip().upper()

def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge mutation values and build mutant sequences, then split into experimental (train_exp.csv) and calc (train_calc.csv)."
    )
    ap.add_argument("--esm2", default="ESM-2_zero-shot_value.csv",
                    help="Path to ESM-2_zero-shot_value.csv (default: %(default)s)")
    ap.add_argument("--ddgf", default="ddGf_value.csv",
                    help="Path to ddGf_value.csv (default: %(default)s)")
    ap.add_argument("--ddgb", default=None,
                    help="Optional path to ddGb_value.csv (omit to skip)")
    ap.add_argument("--wt-seq", default=None, required=True,
                    help="Wild-type amino-acid sequence (string).")
    ap.add_argument("--train_data", "--train-data", dest="train_data", default=None,
                    help="CSV with two columns: mutation ID (e.g., L10A) in col1 and activity in col2.")
    return ap.parse_args()

def load_esm2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    if df.shape[1] < 3:
        df = pd.read_csv(path, header=None, usecols=[1, 2], names=['mutation', 'ESM-2_zero-shot'])
    else:
        df = df.iloc[:, [1, 2]].copy()
        df.columns = ['mutation', 'ESM-2_zero-shot']
    df = df.groupby('mutation', as_index=False)['ESM-2_zero-shot'].mean()
    return df

def load_ddg_generic(path: Path, colname: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    if df.shape[1] < 2:
        df = pd.read_csv(path, header=None, usecols=[0, 1], names=['mutation', colname])
    else:
        df = df.iloc[:, [0, 1]].copy()
        df.columns = ['mutation', colname]
    df = df.groupby('mutation', as_index=False)[colname].mean()
    return df

def apply_single_mutation(wt: str, mut: str) -> str:
    """
    mut: like 'L1A' (1-based index).
    """
    m = MUT_PAT.match(mut)
    if not m:
        raise ValueError(f"Invalid mutation format: {mut}")
    ref, pos_str, alt = m.groups()
    pos = int(pos_str)
    if not (1 <= pos <= len(wt)):
        raise ValueError(f"Position {pos} out of range for WT of length {len(wt)} in mut {mut}")
    wt_ref = wt[pos - 1]
    if wt_ref != ref:
        print(f"Warning: WT at {pos} is {wt_ref}, but mutation expects {ref} ({mut})")
    return wt[:pos - 1] + alt + wt[pos:]

def main():
    args = parse_args()
    wt_seq = args.wt_seq.strip().upper()

    esm2_df = load_esm2(Path(args.esm2))
    ddgf_df = load_ddg_generic(Path(args.ddgf), 'ddGf_rosetta')

    merged = pd.merge(esm2_df, ddgf_df, on='mutation', how='outer')

    has_ddgb = False
    if args.ddgb:
        ddgb_df = load_ddg_generic(Path(args.ddgb), 'ddGb_rosetta')
        merged = pd.merge(merged, ddgb_df, on='mutation', how='outer')
        has_ddgb = True

    def build_seq_safe(mut: str) -> Optional[str]:
        try:
            return apply_single_mutation(wt_seq, str(mut))
        except Exception as e:
            print(f"Skipping {mut}: {e}")
            return None

    merged.insert(0, 'sequence', merged['mutation'].map(build_seq_safe))

    calc_cols = ['sequence', 'ESM-2_zero-shot', 'ddGf_rosetta']
    if has_ddgb:
        calc_cols.append('ddGb_rosetta')

    if args.train_data:
        td = pd.read_csv(args.train_data, header=0)
        if td.shape[1] < 2:
            td = pd.read_csv(args.train_data, header=None, usecols=[0, 1],
                             names=['mutation', 'activity'])
        else:
            td = td.iloc[:, [0, 1]].copy()
            td.columns = ['mutation', 'activity']
        td['mutation'] = td['mutation'].astype(str)

        td.insert(0, 'sequence', td['mutation'].map(build_seq_safe))

        score_cols = [c for c in merged.columns if c in ['ESM-2_zero-shot', 'ddGf_rosetta', 'ddGb_rosetta']]
        exp_join = pd.merge(td, merged[['mutation'] + score_cols], on='mutation', how='left')

        exp_out_cols = ['sequence', 'activity', 'ESM-2_zero-shot', 'ddGf_rosetta']
        if has_ddgb:
            exp_out_cols.append('ddGb_rosetta')
        exp_out = exp_join[exp_out_cols]
        exp_out.to_csv("train_exp.csv", index=False)
        print("Wrote: train_exp.csv")

        td_set = set(td['mutation'].unique())
        calc_df = merged[~merged['mutation'].isin(td_set)].copy()
        calc_out = calc_df[calc_cols]
        calc_out.to_csv("train_calc.csv", index=False)
        print("Wrote: train_calc.csv")
    else:
        merged[calc_cols].to_csv("train_calc.csv", index=False)
        print("Wrote: train_calc.csv")

if __name__ == "__main__":
    main()

