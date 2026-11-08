import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src.config_initialisation import CONFIG
from src.repo_paths import PATHS
from src.utils.amino_acid_information_tools import has_supported_length, is_standard_sequence, substitute
from src.utils.constants import STANDARD_AMINO_ACIDS


# Shipped alongside the full release: the 173 Swiss-Prot entries the bundled FireProtDB file
# refers to, so --prepare runs on a fresh clone. See PREPARE.md.
BUNDLED_SUBSET_NAME = "uniprot_sprot_subset.fasta"


@lru_cache(maxsize=1)
def load_uniprot_sequences(fasta_path: Path) -> dict[str, str]:
    """Map UniProt accessions to sequences; cached because Swiss-Prot is large and reused per source file.

    The full Swiss-Prot release is 275 MB and is not redistributed here, so a clone will not have
    it. When it is absent the bundled subset is used instead: it holds exactly the entries the
    supplied FireProtDB file needs, copied unchanged from the release, so the curation it produces
    is identical. Any other source dataset will refer to accessions the subset does not contain,
    and needs the full release.
    """
    if not fasta_path.is_file():
        subset = fasta_path.with_name(BUNDLED_SUBSET_NAME)
        if not subset.is_file():
            raise FileNotFoundError(
                f"UniProt Swiss-Prot FASTA was not found: {fasta_path}\n"
                f"Neither was the bundled subset: {subset}\n"
                "See PREPARE.md for where to download the full release."
            )
        print(f"NOTE: {fasta_path.name} not found; using the bundled {BUNDLED_SUBSET_NAME}, which covers "
              "only the accessions the supplied FireProtDB file needs. A different source dataset "
              "needs the full Swiss-Prot release (see PREPARE.md).")
        fasta_path = subset
    sequences: dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        accession = record.id.split("|")[1] if "|" in record.id else record.id
        sequences[accession] = str(record.seq).upper()
    return sequences


def pre_cleaner(df: pd.DataFrame, uniprot_path: Path | str = PATHS.uniprot_sprot_fasta) -> pd.DataFrame:
    """Convert FireProtDB source records into the model's standard schema."""
    
    source_columns = ["UNIPROTKB", "SUBSTITUTION", "DDG", "PH", "EXP_TEMPERATURE"]

    missing_columns = set(source_columns).difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Source data is missing required source columns: {sorted(missing_columns)}"
        )

    cleaned = df[source_columns].copy()
    cleaned = cleaned.dropna(subset=source_columns)
    cleaned = cleaned.drop_duplicates(ignore_index=True)

    sequences = load_uniprot_sequences(Path(uniprot_path).resolve())

    substitution_pattern = re.compile(r"^([A-Z])([1-9][0-9]*)([A-Z])$")
    converted_rows: list[dict[str, object]] = []
    discarded_rows = 0
    multiple_mutation_rows = 0

    for _, row in cleaned.iterrows():
        accession = str(row["UNIPROTKB"]).strip()
        substitution = str(row["SUBSTITUTION"]).strip().upper()
        if "," in substitution:
            discarded_rows += 1
            multiple_mutation_rows += 1
            continue

        match = substitution_pattern.fullmatch(substitution)
        sequence = sequences.get(accession)

        if (
            match is None
            or sequence is None
            or not has_supported_length(sequence, CONFIG.max_sequence_length)
            or not is_standard_sequence(sequence)
        ):
            discarded_rows += 1
            continue

        wt_residue, one_indexed_position, mut_residue = match.groups()
        position = int(one_indexed_position) - 1
        if (
            position >= len(sequence)
            or sequence[position] != wt_residue
            or wt_residue == mut_residue
            or mut_residue not in STANDARD_AMINO_ACIDS
        ):
            # Numbering offsets, synonymous or non-standard substitutions would otherwise
            # make run_verification abort the whole preparation step.
            discarded_rows += 1
            continue

        converted_rows.append(
            {
                "sequence": sequence,
                "mutant_sequence": substitute(sequence, position, mut_residue),
                "pH": row["PH"],
                "zero_indexed_mutation_position": position,
                "wt_residue": wt_residue,
                "mut_residue": mut_residue,
                "ddG": row["DDG"],
                "experiment_temperature": pd.to_numeric(
                    row["EXP_TEMPERATURE"], errors="coerce"
                ) + 273.15,
            }
        )

    other_discarded_rows = discarded_rows - multiple_mutation_rows
    if other_discarded_rows:
        print(
            f"NOTE: Discarded {other_discarded_rows} source data rows that were malformed, "
            "out of sequence bounds, inconsistent with the Swiss-Prot residue, or lacked a Swiss-Prot sequence."
        )
    if multiple_mutation_rows:
        print(
            f"NOTE: Discarded {multiple_mutation_rows} source data rows containing "
            "multiple mutations."
        )

    if not converted_rows:
        raise ValueError(
            "No usable rows remained after pre-cleaning this source file: every row was malformed, "
            "carried multiple mutations, or had no matching Swiss-Prot sequence. Check that the file "
            "is FireProtDB data and that data/Uniprot to sequence/uniprot_sprot.fasta is the right release."
        )

    result = pd.DataFrame(converted_rows)
    result = result.dropna(
        subset=[
            "sequence",
            "mutant_sequence",
            "pH",
            "zero_indexed_mutation_position",
            "wt_residue",
            "mut_residue",
            "ddG",
            "experiment_temperature",
        ]
    )
    return result.drop_duplicates(ignore_index=True)
