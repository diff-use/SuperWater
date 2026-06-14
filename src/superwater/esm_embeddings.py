"""In-process generation of ESM-2 per-residue embeddings.

The score/confidence models consume per-residue ESM-2 (``esm2_t33_650M_UR50D``,
layer 33, 1280-dim) embeddings, one ``<name>_chain_<i>.pt`` file per protein chain,
each holding ``{'representations': {33: tensor[L, 1280]}}`` -- the same format that
Meta's ``esm/scripts/extract.py`` produces. Generating them here (via the ``fair-esm``
package) removes the need to clone that repo for a single prediction.
"""
import os

import torch
from Bio.PDB import PDBParser

MODEL_NAME = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
EMBED_DIM = 1280
# Matches the README/extract.py setting used to train the shipped models.
TRUNCATION_SEQ_LENGTH = 4096

# MSE is selenomethionine: chemically almost identical to MET (S replaced by Se).
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M',
    'PHE': 'F', 'PRO': 'P', 'PYL': 'O', 'SER': 'S', 'SEC': 'U', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'XLE': 'J',
}


def get_chain_sequences(pdb_path):
    """Return one amino-acid sequence per chain, in PDB chain order.

    A residue is treated as an amino acid only if it has the CA/N/C backbone atoms;
    waters and other heteroatoms are skipped. Empty strings are kept for non-protein
    chains so that chain indices stay aligned with the receptor graph builder.
    """
    from superwater.structure_io import parse_structure
    structure = parse_structure(pdb_path, permissive=True, structure_id='protein')[0]
    sequences = []
    for chain in structure:
        seq = ''
        for residue in chain:
            if residue.get_resname() == 'HOH':
                continue
            atom_names = {atom.name for atom in residue}
            if {'CA', 'N', 'C'} <= atom_names:
                seq += THREE_TO_ONE.get(residue.get_resname(), '-')
        sequences.append(seq)
    return sequences


def load_esm_model(device):
    """Load (and cache to ~/.cache/torch) the ESM-2 650M model and its alphabet."""
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    return model.to(device).eval(), alphabet


def _embed_sequence(seq, model, alphabet, device, truncation=TRUNCATION_SEQ_LENGTH):
    batch_converter = alphabet.get_batch_converter(truncation)
    _, _, tokens = batch_converter([("protein", seq)])
    tokens = tokens.to(device)
    with torch.inference_mode():
        out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    length = min(truncation, len(seq))
    # Drop the leading BOS token; keep one representation per residue.
    return out["representations"][REPR_LAYER][0, 1:length + 1].cpu().clone()


def embed_complex(name, pdb_path, out_dir, model, alphabet, device, truncation=TRUNCATION_SEQ_LENGTH):
    """Write ``<name>_chain_<i>.pt`` embedding files for every chain of one protein."""
    os.makedirs(out_dir, exist_ok=True)
    for i, seq in enumerate(get_chain_sequences(pdb_path)):
        emb = _embed_sequence(seq, model, alphabet, device, truncation) if seq else torch.zeros(0, EMBED_DIM)
        torch.save({'representations': {REPR_LAYER: emb}}, os.path.join(out_dir, f"{name}_chain_{i}.pt"))


def embed_dataset(data_dir, out_dir, device, truncation=TRUNCATION_SEQ_LENGTH,
                  skip_existing=False):
    """Embed every complex in an organized dataset dir (each subfolder is one complex).

    With ``skip_existing=True``, complexes that already have a ``<name>_chain_0.pt`` in
    ``out_dir`` are left untouched, so a re-run only embeds new/missing complexes.
    """
    from superwater.structure_io import candidate_structure_paths
    names = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    model = alphabet = None  # loaded lazily so a fully-skipped re-run does no model load
    skipped = []
    n_exists = 0
    for name in names:
        if skip_existing and os.path.exists(os.path.join(out_dir, f"{name}_chain_0.pt")):
            n_exists += 1
            continue
        candidates = candidate_structure_paths(os.path.join(data_dir, name), name, "_protein_processed")
        if not candidates:
            reason = f"_protein_processed not found in {os.path.join(data_dir, name)}"
            print(f"[SKIP] {name}: {reason}")
            skipped.append((name, reason))
            continue
        if model is None:
            model, alphabet = load_esm_model(device)
        print(f"Embedding {name} ...")
        last_exc = None
        embedded = False
        for pdb_path in candidates:  # CIF first, fall back to PDB if it fails to parse
            try:
                embed_complex(name, pdb_path, out_dir, model, alphabet, device, truncation)
                embedded = True
                break
            except Exception as exc:
                last_exc = exc
        if not embedded:
            print(f"[SKIP] {name}: embedding error — {last_exc}")
            skipped.append((name, str(last_exc)))
    log_path = os.path.join(data_dir, "logs", "skipped_embedding_errors.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        for name, reason in skipped:
            f.write(f"{name}\t{reason}\n")
    if n_exists:
        print(f"Skipped {n_exists} complexes that already had embeddings")
    if skipped:
        print(f"Wrote {len(skipped)} skipped entries to {log_path}")
