import numpy as np
import warnings
from Bio.PDB import PDBParser


def find_real_water_pos(file_path, model_index=0):
    """Return the (N, 3) array of water-oxygen coordinates from a PDB or CIF/mmCIF file.

    Only PDB and CIF/mmCIF are supported; the pipeline reads waters from the
    ``*_water.{pdb,cif}`` files written during dataset prep. (The legacy ``.mol2`` path,
    which required OpenBabel, has been removed.)
    """
    file_extension = file_path.split('.')[-1].lower()
    if file_extension not in ('pdb', 'cif', 'mmcif'):
        raise ValueError(
            f"Unsupported file format '{file_extension}'. Please provide a PDB or CIF/mmCIF file."
        )

    warnings.simplefilter('ignore')
    if file_extension == 'pdb':
        structure = PDBParser(QUIET=True).get_structure('PDB_structure', file_path)
    else:
        from Bio.PDB.MMCIFParser import MMCIFParser
        structure = MMCIFParser(QUIET=True).get_structure('CIF_structure', file_path)

    water_positions = []
    first_model = next(structure.get_models())
    for chain in first_model:
        for residue in chain:
            for atom in residue:
                if atom.element == 'O':
                    water_positions.append(atom.coord)

    return np.array(water_positions)
