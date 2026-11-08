"""Cleaning and validation for experimental PKAD pKa records.

Each PKAD residue is placed on its chain sequence using the PDB entry's own residue numbering
(mmCIF ``_pdbx_poly_seq_scheme``). Rows that do not follow that numbering fall back to
``align_residue_numbering``, and anything neither step can confirm is dropped.
"""

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm

from src.config_initialisation import CONFIG
from src.data_processing.clustering import cluster_sequences
from src.repo_paths import PATHS
from src.utils.amino_acid_information_tools import has_supported_length, is_standard_sequence
from src.utils.dataframe_processing_tools import read_dataframe, save_dataframe

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _clean_numeric(value: object) -> float | None:
    text = str(value).strip()
    if not text or re.search(r"\b(?:approx|approximately|less than|more than)\b|[<>~]", text, re.I):
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


MAX_NUMBERING_OFFSET = 100  # largest shift (in residues) tried between PDB numbering and the chain sequence
OFFSET_CHANGE_PENALTY = 4  # a new offset must explain more than this many extra residues to be adopted


def align_residue_numbering(
    sequence: str,
    entries: list[tuple[int, str]],
    max_offset: int = MAX_NUMBERING_OFFSET,
    change_penalty: int = OFFSET_CHANGE_PENALTY,
) -> list[int | None]:
    """Map PDB residue numbers ("Res ID") to zero-based positions in the chain sequence.

    PDB numbering often differs from the sequence index by a constant offset, or by an offset
    that changes part-way along the chain (insertion codes, gaps, renumbered tags). Entries are
    walked in residue-number order, keeping the current offset unless switching makes more
    residues match by a clear margin (a small dynamic programme). Each offset is tested by
    checking that the residue letter agrees with the sequence.

    Returns the matched sequence index for each entry (in the input order), or None when no
    consistent offset places that residue on a matching letter. Starting at offset 0 costs
    nothing, so chains that already line up are never moved.
    """
    if not entries:
        return []
    offsets = range(-max_offset, max_offset + 1)
    order = sorted(range(len(entries)), key=lambda index: entries[index][0])

    def mismatch(entry_index: int, offset: int) -> int:
        position = entries[entry_index][0] - 1 + offset
        return 0 if 0 <= position < len(sequence) and sequence[position] == entries[entry_index][1] else 1

    # cost[offset] = fewest mismatches so far + penalties, for paths ending at this offset.
    cost = {offset: (0 if offset == 0 else change_penalty) + mismatch(order[0], offset) for offset in offsets}
    back_pointers: list[dict[int, int]] = []
    for entry_index in order[1:]:
        best_previous = min(cost, key=lambda offset: (cost[offset], abs(offset)))
        new_cost, pointers = {}, {}
        for offset in offsets:
            stay, switch = cost[offset], cost[best_previous] + change_penalty
            pointers[offset] = offset if stay <= switch else best_previous
            new_cost[offset] = min(stay, switch) + mismatch(entry_index, offset)
        cost = new_cost
        back_pointers.append(pointers)

    offset = min(cost, key=lambda candidate: (cost[candidate], abs(candidate)))
    chosen = [offset]
    for pointers in reversed(back_pointers):
        offset = pointers[offset]
        chosen.append(offset)
    chosen.reverse()

    positions: list[int | None] = [None] * len(entries)
    for entry_index, offset in zip(order, chosen):
        if not mismatch(entry_index, offset):
            positions[entry_index] = entries[entry_index][0] - 1 + offset
    return positions


# Exact residue numbering from the PDB entry itself.
@dataclass(frozen=True)
class ChainNumbering:
    """A chain's canonical sequence and the PDB residue numbering of each sequence position."""

    sequence: str
    positions: dict[tuple[int, str], int]  # (PDB residue number, insertion code) -> zero-based index


@dataclass(frozen=True)
class PkadRow:
    pdb_id: str
    chain: str
    residue: str
    number: int
    insertion_code: str
    pka: float


def _as_list(value: str | list[str]) -> list[str]:
    """mmCIF items are plain strings when a category has one row, and lists otherwise."""
    return value if isinstance(value, list) else [value]


def parse_mmcif_numbering(mmcif_text: str) -> dict[str, ChainNumbering]:
    """Read every polymer chain's sequence and residue-number mapping from an mmCIF file.

    ``_pdbx_poly_seq_scheme`` lists each position of the deposited sequence (``seq_id``) with
    the author chain (``pdb_strand_id``), the PDB-format residue number (``pdb_seq_num``, set
    even for residues not seen in the density) and insertion code, i.e. the numbering PKAD's
    "Res ID" refers to. The sequence is ``_entity_poly.pdbx_seq_one_letter_code_can``, the same
    canonical sequence the RCSB FASTA download returns.
    """
    data = MMCIF2Dict(StringIO(mmcif_text))
    sequences = {
        entity: re.sub(r"\s+", "", code).upper()
        for entity, code in zip(
            _as_list(data["_entity_poly.entity_id"]),
            _as_list(data["_entity_poly.pdbx_seq_one_letter_code_can"]),
        )
    }
    columns = ("pdb_strand_id", "entity_id", "seq_id", "pdb_seq_num", "pdb_ins_code")
    scheme = zip(*(_as_list(data[f"_pdbx_poly_seq_scheme.{column}"]) for column in columns))

    chains: dict[str, ChainNumbering] = {}
    for chain, entity, seq_id, number, insertion_code in scheme:
        if entity not in sequences or not re.fullmatch(r"-?\d+", number):
            continue
        record = chains.setdefault(chain, ChainNumbering(sequences[entity], {}))
        key = (int(number), "" if insertion_code in (".", "?") else insertion_code.upper())
        # Microheterogeneity lists one position twice; the first entry is kept.
        record.positions.setdefault(key, int(seq_id) - 1)
    return chains


def _fetch_entry_numbering(pdb_id: str) -> dict[str, ChainNumbering]:
    response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif", timeout=60)
    response.raise_for_status()
    return parse_mmcif_numbering(response.text)


def _find_chain(chains: dict[str, ChainNumbering], chain: str) -> ChainNumbering | None:
    if chain in chains:
        return chains[chain]
    matches = [record for name, record in chains.items() if name.upper() == chain.upper()]
    return matches[0] if len(matches) == 1 else None


def place_chain_residues(numbering: ChainNumbering, rows: list[PkadRow]) -> tuple[list[int | None], int]:
    """Place each PKAD row on a sequence index; returns (indices, number placed by the fallback).

    1. Exact: look up (Res ID, insertion code) in the PDB's own numbering and confirm the letter.
    2. Fallback, only for rows that fail step 1 (e.g. PKAD used the original paper's numbering):
       ``align_residue_numbering`` on those rows, never reusing a position already placed exactly.
    Rows that neither step can confirm stay None and are dropped.
    """
    sequence = numbering.sequence
    indices: list[int | None] = []
    for row in rows:
        index = numbering.positions.get((row.number, row.insertion_code))
        indices.append(index if index is not None and sequence[index] == row.residue else None)

    unplaced = [position for position, index in enumerate(indices) if index is None]
    if not unplaced:
        return indices, 0
    taken = {index for index in indices if index is not None}
    fallback = align_residue_numbering(sequence, [(rows[i].number, rows[i].residue) for i in unplaced])
    recovered = 0
    for row_position, index in zip(unplaced, fallback):
        if index is not None and index not in taken:
            indices[row_position] = index
            taken.add(index)
            recovered += 1
    return indices, recovered


def _parse_residue_number(value: object) -> tuple[int, str] | None:
    """Split a PKAD "Res ID" such as ``52``, ``52.0`` or ``52A`` into (number, insertion code)."""
    match = re.fullmatch(r"\s*(-?\d+)(?:\.0+)?\s*([A-Za-z]?)\s*", str(value))
    return (int(match.group(1)), match.group(2).upper()) if match else None


def _read_pkad_rows(frame: pd.DataFrame) -> list[PkadRow]:
    """Keep rows with a valid PDB ID, chain, standard residue, residue number and exact pKa."""
    rows = []
    for _, record in frame.iterrows():
        pdb_id = str(record["PDB ID"]).strip().upper()
        chain = str(record["Chain"]).strip()
        residue = THREE_TO_ONE.get(str(record["Res Name"]).strip().upper())
        residue_number = _parse_residue_number(record["Res ID"])
        pka = _clean_numeric(record["Expt. pKa"])
        if re.fullmatch(r"[0-9A-Z]{4}", pdb_id) and chain and residue and residue_number and pka is not None:
            rows.append(PkadRow(pdb_id, chain, residue, *residue_number, pka))
    return rows


def clean_pkad_data(source_path: Path | str = PATHS.pkad_source_csv) -> pd.DataFrame:
    """Clean PKAD records, placing each residue with the PDB entry's own residue numbering."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"PKAD source file was not found: {source}")
    frame = pd.read_csv(source, dtype={"Res ID": str, "Chain": str})
    required = {"PDB ID", "Res Name", "Chain", "Res ID", "Expt. pKa"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"PKAD source is missing columns: {sorted(missing)}")

    rows_by_chain: dict[tuple[str, str], list[PkadRow]] = {}
    for row in _read_pkad_rows(frame):
        rows_by_chain.setdefault((row.pdb_id, row.chain), []).append(row)

    entries: dict[str, dict[str, ChainNumbering]] = {}
    failed_entries: list[str] = []
    for pdb_id in tqdm(sorted({pdb_id for pdb_id, _ in rows_by_chain}), desc="Fetching PKAD mmCIF files", unit="entry"):
        try:
            entries[pdb_id] = _fetch_entry_numbering(pdb_id)
        except (requests.RequestException, KeyError, ValueError):
            failed_entries.append(pdb_id)

    records: list[dict[str, object]] = []
    counts = {"exact": 0, "fallback": 0, "dropped": 0, "unusable_chain": 0}
    fallback_chains: list[str] = []
    for (pdb_id, chain), rows in rows_by_chain.items():
        numbering = _find_chain(entries.get(pdb_id, {}), chain)
        if (
            numbering is None
            or not is_standard_sequence(numbering.sequence)
            or not has_supported_length(numbering.sequence, CONFIG.max_sequence_length)
        ):
            counts["unusable_chain"] += len(rows)
            continue

        indices, recovered = place_chain_residues(numbering, rows)
        placed = sum(index is not None for index in indices)
        counts["exact"] += placed - recovered
        counts["fallback"] += recovered
        counts["dropped"] += len(rows) - placed
        if recovered:
            fallback_chains.append(f"{pdb_id}:{chain}")
        records.extend(
            {
                "pdb_id": pdb_id,
                "chain": chain,
                "sequence": numbering.sequence,
                "residue_position": index,
                "residue": row.residue,
                "pKa": row.pka,
            }
            for row, index in zip(rows, indices)
            if index is not None
        )

    cleaned = pd.DataFrame(
        records, columns=["pdb_id", "chain", "sequence", "residue_position", "residue", "pKa"]
    ).drop_duplicates(ignore_index=True)
    if cleaned.empty:
        raise ValueError("PKAD cleaning produced no verified records")
    # One target per residue: repeat measurements of the same residue are averaged.
    cleaned = (
        cleaned.groupby(["sequence", "residue_position", "residue"], as_index=False, sort=False)
        .agg(pKa=("pKa", "mean"), pdb_id=("pdb_id", "first"), chain=("chain", "first"))
        [["pdb_id", "chain", "sequence", "residue_position", "residue", "pKa"]]
    )

    print(
        f"PKAD numbering: {counts['exact']} rows placed exactly from the PDB numbering, "
        f"{counts['fallback']} by the fallback alignment, {counts['dropped']} dropped; "
        f"{counts['unusable_chain']} rows skipped (chain missing, non-standard or outside length limits); "
        f"{len(failed_entries)} entries could not be downloaded or read."
    )
    if fallback_chains:
        print(f"Chains whose numbering does not follow the PDB file (fallback used): {', '.join(fallback_chains)}")
    if failed_entries:
        print(f"Entries not retrieved: {', '.join(failed_entries)}")
    return cleaned


def remove_testing_set_similar_sequences(
    pkad_frame: pd.DataFrame,
    validation_path: Path | str,
) -> pd.DataFrame:
    """Discard PKAD sequences clustered at or above the DDG validation identity threshold."""
    validation = read_dataframe(validation_path)
    if "sequence" not in validation.columns:
        raise ValueError(f"Validation data {validation_path} has no sequence column")
    clustered = cluster_sequences(pd.concat([pkad_frame["sequence"], validation["sequence"]]))
    validation_clusters = set(
        clustered.loc[clustered["sequence"].isin(validation["sequence"]), "cluster_representative"]
    )
    blocked_sequences = set(
        clustered.loc[clustered["cluster_representative"].isin(validation_clusters), "sequence"]
    )
    return pkad_frame[~pkad_frame["sequence"].isin(blocked_sequences)].reset_index(drop=True)


def prepare_pkad_data(output_path: Path | str = PATHS.pkad_cleaned_csv) -> pd.DataFrame:
    """Clean PKAD and save every usable record.

    Nothing is filtered out here. Which PKAD chains have to be withheld depends on which ddG
    proteins are in the testing set, and that differs from one training split to the next, so
    validation training filters this table per split. The model trained for use holds nothing
    out and therefore uses all of it.
    """
    pkad = clean_pkad_data()
    if pkad.empty:
        raise ValueError("No usable PKAD records remain after cleaning")
    save_dataframe(output_path, pkad)
    return pkad
