import numpy as np
import warnings
from Bio.PDB import PDBParser

def find_real_water_pos(file_path, model_index=0):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension in ('pdb', 'cif', 'mmcif'):
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


    elif file_extension == 'mol2':
        # Lazy import: OpenBabel is only needed for the (now optional) mol2 path, so the
        # package no longer hard-requires it.
        from openbabel import openbabel as ob
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("mol2")
        mol = ob.OBMol()
        obConversion.ReadFile(mol, file_path)
        water_positions = []
        for atom in ob.OBMolAtomIter(mol):
            # Match oxygen by atomic number (8); OpenBabel reports Sybyl types such
            # as 'O3' for mol2 water, so comparing GetType() to 'O' never matches.
            if atom.GetAtomicNum() == 8:
                water_positions.append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()]))

    else:
        raise ValueError("Unsupported file format. Please provide a PDB, CIF/mmCIF, or MOL2 file.")
        
    return np.array(water_positions)