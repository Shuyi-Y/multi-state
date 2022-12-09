import os
import json
from pprint import pprint


import Bio
import pandas as pd
import numpy as np
import torch

from typing import List, Mapping, Optional, Dict
from absl import logging
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
# from Bio.PDB import Atom, Chain, Model, Entity, Residue, Structure

# import utils
# from protutils.protein import Chain


charactres = ['C', 'D', 'S', 'Q', 'K', 'I', 'P', 
              'T', 'F', 'N', 'G', 'H', 'L', 'R', 
              'W', 'A', 'V', 'E', 'Y', 'M', 'U', 'O', 'X']
aa_types = ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO',
            'THR', 'PHE', 'ASN', 'GLY', 'HIS', 'LEU', 'ARG',
            'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET', 'SEC', 'PYL', 'UNK']

chars_to_aa_types = {c: aa for c, aa in zip(charactres, aa_types)}
aa_types_to_chars = {aa: c for c, aa in zip(charactres, aa_types)}

atomtypes = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1",
    "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1",
    "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
restype3_to_atoms = {
    "ALA": [["N", 0, [-0.525, 1.363, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.526, -0.0, -0.0]],
            ["O", 3, [0.627, 1.062, 0.0]], ["CB", 0, [-0.529, -0.774, -1.205]]],
    "ARG": [["N", 0, [-0.524, 1.362, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.525, -0.0, -0.0]],
            ["O", 3, [0.626, 1.062, 0.0]], ["CB", 0, [-0.524, -0.778, -1.209]], ["CG", 4, [0.616, 1.39, -0.0]],
            ["CD", 5, [0.564, 1.414, 0.0]], ["NE", 6, [0.539, 1.357, -0.0]], ["CZ", 7, [0.758, 1.093, -0.0]],
            ["NH1", 7, [0.206, 2.301, 0.0]], ["NH2", 7, [2.078, 0.978, -0.0]]],
    "ASN": [["N", 0, [-0.536, 1.357, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.526, -0.0, -0.0]],
            ["O", 3, [0.625, 1.062, 0.0]], ["CB", 0, [-0.531, -0.787, -1.2]], ["CG", 4, [0.584, 1.399, 0.0]],
            ["OD1", 5, [0.633, 1.059, 0.0]], ["ND2", 5, [0.593, -1.188, 0.001]]],
    "ASP": [["N", 0, [-0.525, 1.362, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.527, 0.0, -0.0]],
            ["O", 3, [0.626, 1.062, -0.0]], ["CB", 0, [-0.526, -0.778, -1.208]], ["CG", 4, [0.593, 1.398, -0.0]],
            ["OD1", 5, [0.61, 1.091, 0.0]], ["OD2", 5, [0.592, -1.101, -0.003]]],
    "CYS": [["N", 0, [-0.522, 1.362, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.524, 0.0, 0.0]],
            ["O", 3, [0.625, 1.062, -0.0]], ["CB", 0, [-0.519, -0.773, -1.212]], ["SG", 4, [0.728, 1.653, 0.0]]],
    "GLN": [["N", 0, [-0.526, 1.361, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.526, 0.0, 0.0]],
            ["O", 3, [0.626, 1.062, -0.0]], ["CB", 0, [-0.525, -0.779, -1.207]], ["CG", 4, [0.615, 1.393, 0.0]],
            ["CD", 5, [0.587, 1.399, -0.0]], ["OE1", 6, [0.634, 1.06, 0.0]], ["NE2", 6, [0.593, -1.189, -0.001]]],
    "GLU": [["N", 0, [-0.528, 1.361, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.526, -0.0, -0.0]],
            ["O", 3, [0.626, 1.062, 0.0]], ["CB", 0, [-0.526, -0.781, -1.207]], ["CG", 4, [0.615, 1.392, 0.0]],
            ["CD", 5, [0.6, 1.397, 0.0]], ["OE1", 6, [0.607, 1.095, -0.0]], ["OE2", 6, [0.589, -1.104, -0.001]]],
    "GLY": [["N", 0, [-0.572, 1.337, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.517, -0.0, -0.0]],
            ["O", 3, [0.626, 1.062, -0.0]]],
    "HIS": [["N", 0, [-0.527, 1.36, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.525, 0.0, 0.0]],
            ["O", 3, [0.625, 1.063, 0.0]], ["CB", 0, [-0.525, -0.778, -1.208]], ["CG", 4, [0.6, 1.37, -0.0]],
            ["ND1", 5, [0.744, 1.16, -0.0]], ["CD2", 5, [0.889, -1.021, 0.003]], ["CE1", 5, [2.03, 0.851, 0.002]],
            ["NE2", 5, [2.145, -0.466, 0.004]]],
    "ILE": [["N", 0, [-0.493, 1.373, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.527, -0.0, -0.0]],
            ["O", 3, [0.627, 1.062, -0.0]], ["CB", 0, [-0.536, -0.793, -1.213]], ["CG1", 4, [0.534, 1.437, -0.0]],
            ["CG2", 4, [0.54, -0.785, -1.199]], ["CD1", 5, [0.619, 1.391, 0.0]]],
    "LEU": [["N", 0, [-0.52, 1.363, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.525, -0.0, -0.0]],
            ["O", 3, [0.625, 1.063, -0.0]], ["CB", 0, [-0.522, -0.773, -1.214]], ["CG", 4, [0.678, 1.371, 0.0]],
            ["CD1", 5, [0.53, 1.43, -0.0]], ["CD2", 5, [0.535, -0.774, 1.2]]],
    "LYS": [["N", 0, [-0.526, 1.362, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.526, 0.0, 0.0]],
            ["O", 3, [0.626, 1.062, -0.0]], ["CB", 0, [-0.524, -0.778, -1.208]], ["CG", 4, [0.619, 1.39, 0.0]],
            ["CD", 5, [0.559, 1.417, 0.0]], ["CE", 6, [0.56, 1.416, 0.0]], ["NZ", 7, [0.554, 1.387, 0.0]]],
    "MET": [["N", 0, [-0.521, 1.364, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.525, 0.0, 0.0]],
            ["O", 3, [0.625, 1.062, -0.0]], ["CB", 0, [-0.523, -0.776, -1.21]], ["CG", 4, [0.613, 1.391, -0.0]],
            ["SD", 5, [0.703, 1.695, 0.0]], ["CE", 6, [0.32, 1.786, -0.0]]],
    "PHE": [["N", 0, [-0.518, 1.363, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.524, 0.0, -0.0]],
            ["O", 3, [0.626, 1.062, -0.0]], ["CB", 0, [-0.525, -0.776, -1.212]], ["CG", 4, [0.607, 1.377, 0.0]],
            ["CD1", 5, [0.709, 1.195, -0.0]], ["CD2", 5, [0.706, -1.196, 0.0]], ["CE1", 5, [2.102, 1.198, -0.0]],
            ["CE2", 5, [2.098, -1.201, -0.0]], ["CZ", 5, [2.794, -0.003, -0.001]]],
    "PRO": [["N", 0, [-0.566, 1.351, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.527, -0.0, 0.0]],
            ["O", 3, [0.621, 1.066, 0.0]], ["CB", 0, [-0.546, -0.611, -1.293]], ["CG", 4, [0.382, 1.445, 0.0]],
            ["CD", 5, [0.477, 1.424, 0.0]]],
    "SER": [["N", 0, [-0.529, 1.36, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.525, -0.0, -0.0]],
            ["O", 3, [0.626, 1.062, -0.0]], ["CB", 0, [-0.518, -0.777, -1.211]], ["OG", 4, [0.503, 1.325, 0.0]]],
    "THR": [["N", 0, [-0.517, 1.364, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.526, 0.0, -0.0]],
            ["O", 3, [0.626, 1.062, 0.0]], ["CB", 0, [-0.516, -0.793, -1.215]], ["OG1", 4, [0.472, 1.353, 0.0]],
            ["CG2", 4, [0.55, -0.718, -1.228]]],
    "TRP": [["N", 0, [-0.521, 1.363, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.525, -0.0, 0.0]],
            ["O", 3, [0.627, 1.062, 0.0]], ["CB", 0, [-0.523, -0.776, -1.212]], ["CG", 4, [0.609, 1.37, -0.0]],
            ["CD1", 5, [0.824, 1.091, 0.0]], ["CD2", 5, [0.854, -1.148, -0.005]], ["NE1", 5, [2.14, 0.69, -0.004]],
            ["CE2", 5, [2.186, -0.678, -0.007]], ["CE3", 5, [0.622, -2.53, -0.007]],
            ["CZ2", 5, [3.283, -1.543, -0.011]],
            ["CZ3", 5, [1.715, -3.389, -0.011]], ["CH2", 5, [3.028, -2.89, -0.013]]],
    "TYR": [["N", 0, [-0.522, 1.362, 0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.524, -0.0, -0.0]],
            ["O", 3, [0.627, 1.062, -0.0]], ["CB", 0, [-0.522, -0.776, -1.213]], ["CG", 4, [0.607, 1.382, -0.0]],
            ["CD1", 5, [0.716, 1.195, -0.0]], ["CD2", 5, [0.713, -1.194, -0.001]], ["CE1", 5, [2.107, 1.2, -0.002]],
            ["CE2", 5, [2.104, -1.201, -0.003]], ["CZ", 5, [2.791, -0.001, -0.003]],
            ["OH", 5, [4.168, -0.002, -0.005]]],
    "VAL": [["N", 0, [-0.494, 1.373, -0.0]], ["CA", 0, [0.0, 0.0, 0.0]], ["C", 0, [1.527, -0.0, -0.0]],
            ["O", 3, [0.627, 1.062, -0.0]], ["CB", 0, [-0.533, -0.795, -1.213]], ["CG1", 4, [0.54, 1.429, -0.0]],
            ["CG2", 4, [0.533, -0.776, 1.203]]],
    "UNK": []
}
restype3_to_atomtypes = {restype3: list(zip(*atomtypes14))[0] if restype3 != "UNK" else ()
                         for restype3, atomtypes14 in restype3_to_atoms.items()}


def pipeline(GN_A_all: pd.DataFrame) -> pd.DataFrame:
    '''Filter redundant items in the GN_A_all.csv'''
    retained_data_indices = []
    
    for idx in range(GN_A_all.shape[0]):
        item = GN_A_all.iloc[idx, :].to_list()[0]
        # print(item)
        if item[0].isdigit():
            retained_data_indices.append(idx)
    
    return GN_A_all.iloc[retained_data_indices]


def filter_GPCR_A_prots(pdb_list: pd.DataFrame, 
                        target_family: str = 'A',
                        is_saved_json: bool = True) -> pd.DataFrame:
    '''Filter those proteins which do not belong to GPCR A family.
    '''
    df_gpcr_A_family = pdb_list[(pdb_list['Cl.'] == target_family)]
    gpcr_A_family = pd.DataFrame(df_gpcr_A_family, columns = ['PDB', 'Cl.'])
    gpcr_A_family_dict = gpcr_A_family.set_index('PDB').to_dict('dict')['Cl.']
    
    if is_saved_json:
        with open('gpcr_A_family.json', 'w+') as f:
            f.write(json.dumps(gpcr_A_family_dict, ensure_ascii = False, indent = 2))
    
    return df_gpcr_A_family


def generate_GN_numbers(pdb_list: pd.DataFrame, 
                        GN_A_all: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    res_to_GN, GN_to_res, ineffective_gpcr = {}, {}, {}
    # 把所有的GPCR中不属于A家族的给过滤掉
    gpcr_A_prots = filter_GPCR_A_prots(pdb_list)
    # 提取所有的GPCR A家族蛋白的PDB IDs
    pdb_ids = gpcr_A_prots['PDB'].to_list()
    # 将GN_A中所有列名里面的空格剔除再送回原处
    GN_A_all.rename(columns = {c: c.strip() for c in GN_A_all.columns.to_list()}, inplace = True)
    # 获取所有可提取目标残基的蛋白在GPCRDB中的名字
    curr_gpcr_prots = GN_A_all.columns.to_list()[1:]
    # 将所有通用残基编号缺失的值改为NAN
    GN_A_all.replace('-', np.nan, inplace = True)
    
    for pdb_id in pdb_ids:
        IUPHAR_name = gpcr_A_prots[gpcr_A_prots['PDB'] == pdb_id]['IUPHAR'].to_list()[0]
        if IUPHAR_name not in curr_gpcr_prots:
            # If the IUPHAR name from pdb_list.csv have no GN numbers
            print(f'We have no GN numbers for the PDB file {pdb_id}.')
            logging.info(f'We have no GN numbers for the PDB file {pdb_id}.')
            
            # Record the not-found pdb files
            # For example: 5-HT4: 7XT8
            ineffective_gpcr[pdb_id] = IUPHAR_name
            continue

        GN_numbers = pd.DataFrame(GN_A_all, columns = ['GPCRdb(A)', IUPHAR_name])
        GN_numbers = GN_numbers.dropna()
        dict_GN_to_res = GN_numbers.set_index('GPCRdb(A)').to_dict('dict')[IUPHAR_name]
        dict_res_to_GN = GN_numbers.set_index(IUPHAR_name).to_dict('dict')['GPCRdb(A)']
        
        res_to_GN[pdb_id] = dict_res_to_GN
        GN_to_res[pdb_id] = dict_GN_to_res
    
    # print(len(res_to_GN.keys()), len(ineffective_gpcr.keys()))  # 526, 86
    with open('PDBID1.json', 'w+') as f:
        f.write(json.dumps(res_to_GN, ensure_ascii = False, indent = 2))
    
    with open('PDBID2.json', 'w+') as f:
        f.write(json.dumps(GN_to_res, ensure_ascii = False, indent = 2))
    
    with open('invalid_PDBID.json', 'w+') as f:
        f.write(json.dumps(ineffective_gpcr, ensure_ascii = False, indent = 2))

    print('The PDBID json file is created successfully!')
    logging.info('The PDBID json file is created successfully!')
    
    return res_to_GN, GN_to_res


def get_coords_from_GN(pdb_file_pth: str, 
                       res_to_GN: Dict,
                       pdb_list: pd.DataFrame,
                       drop_rate: float = 0.1) -> torch.Tensor:
    if not os.path.exists(pdb_file_pth):
        raise ValueError('The directory does not exist!')
    # Obtain the name of the given PDB or MMCIF file 
    # such as: 1F88.pdb
    pdb_fname = os.path.basename(pdb_file_pth)

    if pdb_fname.endswith('.cif') or pdb_fname.endswith('.CIF'):
        parser = MMCIFParser()
    if pdb_fname.endswith('.pdb') or pdb_fname.endswith('.PDB'):
        parser =PDBParser()
    else:
        raise ValueError('This file is not a PDB or MMCIF file!')
    
    pdb_id = pdb_fname.split('.')[0][ :4]
    structure = parser.get_structure(pdb_id, pdb_file_pth)
    models = list(structure.get_models())
    # If the number od models is not 1
    # raise error
    if len(models) > 1:
        print(f'Only PDB file with single model can be processed!')
        return (None, None)
    model = models[0]
    # Find the GN numbers corresponding to the PDB ID
    dict_res_to_GN = res_to_GN[pdb_id]
    # num_selected_res = len(list(dict_res_to_GN.keys()))
    # Find the preferred chain for the given PDB ID in the pdb_list
    preferred_chains = pd.DataFrame(pdb_list, columns = ['PDB', 'Preferred chain']).set_index('PDB').to_dict('dict')['Preferred chain']
    target_chain_id = preferred_chains[pdb_id]
    chain = model[target_chain_id]
    # Find all residues of the preferred chain
    residues = list(chain.get_residues())
    num_res = len(residues)
    
    GN_res_list = list(dict_res_to_GN.keys())
    mismatch_res_count = 0
    all_res_atom_coords, all_res_GN_num = [], []
    
    aatype_list, pos_list = [], []
    for GN_res in GN_res_list:
        aa, p = GN_res[0], GN_res[1: ]
        aatype_list.append(aa)
        pos_list.append(p)
    
    dict_mismatch_res = {}
    
    # Reverse all residues in the preferred chain
    for res in residues:
        # For example: GLU, 80
        aatype, pos = res.resname, res.get_id()[1]
        # some residues, such as 8K3, CLR.
        # They are not common aimo acid, so drop them.
        if aatype not in aa_types:
            continue
        # When we find the current pos does not exist, skip.
        if str(pos) not in pos_list:
            continue
        # If the pos exists but the aatype does not match, this match is wrong.
        # Count once, and check the next residue
        expected_restype = chars_to_aa_types[aatype_list[pos_list.index(str(pos))]]
        if expected_restype != aatype:
            mismatch_res_count += 1
            dict_mismatch_res[expected_restype+pos_list[pos_list.index(str(pos))]] = res.resname + str(pos)
            continue
        
        res_atom_coords = torch.zeros((14, 3), dtype = torch.float)
        for atom in res:
            if atom.name not in atomtypes[:36]: # OXT ignored
                continue
            atom_idx = restype3_to_atomtypes[aatype].index(atom.name)
            res_atom_coords[atom_idx] = torch.from_numpy(atom.coord)
            # print(res_atom_coords)
        all_res_atom_coords.append(res_atom_coords)
        all_res_GN_num.append(dict_res_to_GN[aa_types_to_chars[aatype] + str(pos)])
    
    # If the ratio of mismatching residues is larger than drop_rate, drop this PDB file.
    print(f'PDB ID: {pdb_id}, mismatch/all: {mismatch_res_count}/{num_res}')
    if float(mismatch_res_count / num_res) >= drop_rate:
        return (None, dict_mismatch_res)

    dict_GN_to_coords = {gn_num: coords.tolist() for gn_num, coords in zip(all_res_GN_num, all_res_atom_coords)}
            
    return dict_GN_to_coords, dict_mismatch_res


def get_coords_for_all_pdb_files(csv_pth: str,
                                 pdb_files_dir: str,
                                 is_saved_json: Optional[bool] = True) -> Dict[str, torch.Tensor]:
    
    pdb_list = pd.read_excel(csv_pth, sheet_name = 'PDB_list', dtype = {'Resolution': np.float16})
    print(f'pdb_list rows: {pdb_list.shape[0]}')  # 885 rows
    logging.info(f'pdb_list rows: {pdb_list.shape[0]}')
    
    pdb_list = filter_GPCR_A_prots(pdb_list)
    print(f'pdb_list rows (processed): {pdb_list.shape[0]}')  # 612 rows
    logging.info(f'pdb_list rows (processed): {pdb_list.shape[0]}')

    # filter some lines starting without a number in GN_A_all.csv (such as TM1, CL1, ...)
    GN_A_all = pd.read_excel(csv_pth, sheet_name = 'GN_A')  # 391 rows
    print(f'GN_A_all rows: {GN_A_all.shape[0]}')  # 379 rows
    GN_A_all = pipeline(GN_A_all)
    print(f'GN_A_all rows (processed): {GN_A_all.shape[0]}')
    
    pdbs = pdb_list['PDB'].to_list()
    # pprint(pdbs)
    pdb_files_pth_list = os.listdir(pdb_files_dir)
    # print(pdb_files_pth_list)
    
    res_to_GN, GN_to_res= generate_GN_numbers(pdb_list, GN_A_all)
    
    dict_all_res_coords, dict_all_mismatch_res = {}, {}
    
    for pdb_file_pth in pdb_files_pth_list:
        pdb_fname = os.path.basename(pdb_file_pth)
        pdb_id = pdb_fname.split('.')[0][ :4]
        
        if pdb_id not in pdbs:
            print(f'The PDB {pdb_id} does not have corresponding item in the source csv file!')
            continue
        
        IUPHAR_name = pdb_list[pdb_list['PDB'] == pdb_id]['IUPHAR'].to_list()[0]
        # 将GN_A中所有列名里面的空格剔除再送回原处
        GN_A_all.rename(columns = {c: c.strip() for c in GN_A_all.columns.to_list()}, inplace = True)
        IUPHAR_list = GN_A_all.columns.to_list()[1: ]
        if IUPHAR_name not in IUPHAR_list:
            print(f'The current PDB {pdb_id} has the IUPHAR name {IUPHAR_name}, but it does not exist in our data!')
            continue
        
        pdb_file_pth = os.path.join(pdb_files_dir, pdb_file_pth)
        dict_GN_to_coords, dict_mismatch_res = get_coords_from_GN(pdb_file_pth, res_to_GN, pdb_list)
        if dict_GN_to_coords is None or dict_mismatch_res is None:
            print(f'The PDB {pdb_id} shoule be dropped!')
            continue
        
        dict_all_res_coords[pdb_id] = dict_GN_to_coords
        dict_all_mismatch_res[pdb_id] = dict_mismatch_res
        
        if is_saved_json:
            with open('coords.json', 'w+') as f:
                f.write(json.dumps(dict_all_res_coords, ensure_ascii = False, indent = 2))
            
            with open('mismatch_res.json', 'w+') as f:
                f.write(json.dumps(dict_all_mismatch_res, ensure_ascii = False, indent = 2))


def get_res_pos(pdb_file_pth: str, 
                res_to_GN: Dict,
                pdb_list: pd.DataFrame,
                drop_rate: float = 0.1):
    if not os.path.exists(pdb_file_pth):
        raise ValueError('The directory does not exist!')
    # Obtain the name of the given PDB or MMCIF file 
    # such as: 1F88.pdb
    pdb_fname = os.path.basename(pdb_file_pth)

    if pdb_fname.endswith('.cif') or pdb_fname.endswith('.CIF'):
        parser = MMCIFParser()
    if pdb_fname.endswith('.pdb') or pdb_fname.endswith('.PDB'):
        parser =PDBParser()
    else:
        raise ValueError('This file is not a PDB or MMCIF file!')
    
    pdb_id = pdb_fname.split('.')[0][ :4]
    structure = parser.get_structure(pdb_id, pdb_file_pth)
    models = list(structure.get_models())
    # If the number od models is not 1
    # raise error
    if len(models) > 1:
        print(f'Only PDB file with single model can be processed!')
        return (None, None)
    model = models[0]
    # Find the GN numbers corresponding to the PDB ID
    dict_res_to_GN = res_to_GN[pdb_id]
    
    # num_selected_res = len(list(dict_res_to_GN.keys()))
    # Find the preferred chain for the given PDB ID in the pdb_list
    preferred_chains = pd.DataFrame(pdb_list, columns = ['PDB', 'Preferred chain']).set_index('PDB').to_dict('dict')['Preferred chain']
    target_chain_id = preferred_chains[pdb_id]
    chain = model[target_chain_id]
    
    # Find all residues of the preferred chain
    residues = list(chain.get_residues())
    num_res = len(residues)
    
    GN_res_list = list(dict_res_to_GN.keys())
    mismatch_res_count = 0
    # all_res_atom_coords, all_res_GN_num = [], []
    
    aatype_list, pos_list = [], []
    for GN_res in GN_res_list:
        aa, p = GN_res[0], GN_res[1: ]
        aatype_list.append(aa)
        pos_list.append(p)
    
    dict_mismatch_res = {}
    dict_GN_to_pos = {}
    
    # Reverse all residues in the preferred chain
    for res in residues:
        # For example: GLU, 80
        aatype, pos = res.resname, res.get_id()[1]
        # some residues, such as 8K3, CLR.
        # They are not common aimo acid, so drop them.
        if aatype not in aa_types:
            continue
        # When we find the current pos does not exist, skip.
        if str(pos) not in pos_list:
            continue
        # If the pos exists but the aatype does not match, this match is wrong.
        # Count once, and check the next residue
        expected_restype = chars_to_aa_types[aatype_list[pos_list.index(str(pos))]]
        if expected_restype != aatype:
            mismatch_res_count += 1
            dict_mismatch_res[expected_restype+pos_list[pos_list.index(str(pos))]] = res.resname + str(pos)
            continue
        
        dict_GN_to_pos[dict_res_to_GN[aa_types_to_chars[aatype] + str(pos)]] = pos
    
    # If the ratio of mismatching residues is larger than drop_rate, drop this PDB file.
    print(f'PDB ID: {pdb_id}, mismatch/all: {mismatch_res_count}/{num_res}')
    if float(mismatch_res_count / num_res) >= drop_rate:
        return (None, mismatch_res_count)

    # dict_GN_to_coords = {gn_num: coords.tolist() for gn_num, coords in zip(all_res_GN_num, all_res_atom_coords)}
            
    return dict_GN_to_pos, dict_mismatch_res


def get_all_res_pose(csv_pth: str,
                     pdb_files_dir: str,
                     is_saved_json: Optional[bool] = True) -> Dict[str, torch.Tensor]:
    
    pdb_list = pd.read_excel(csv_pth, sheet_name = 'PDB_list', dtype = {'Resolution': np.float16})
    print(f'pdb_list rows: {pdb_list.shape[0]}')  # 885 rows
    logging.info(f'pdb_list rows: {pdb_list.shape[0]}')
    
    pdb_list = filter_GPCR_A_prots(pdb_list)
    print(f'pdb_list rows (processed): {pdb_list.shape[0]}')  # 612 rows
    logging.info(f'pdb_list rows (processed): {pdb_list.shape[0]}')

    # filter some lines starting without a number in GN_A_all.csv (such as TM1, CL1, ...)
    GN_A_all = pd.read_excel(csv_pth, sheet_name = 'GN_A')  # 391 rows
    print(f'GN_A_all rows: {GN_A_all.shape[0]}')  # 379 rows
    GN_A_all = pipeline(GN_A_all)
    print(f'GN_A_all rows (processed): {GN_A_all.shape[0]}')
    
    pdbs = pdb_list['PDB'].to_list()
    # pprint(pdbs)
    pdb_files_pth_list = os.listdir(pdb_files_dir)
    # print(pdb_files_pth_list)
    
    res_to_GN, GN_to_res= generate_GN_numbers(pdb_list, GN_A_all)
    
    dict_all_res_pos, dict_all_mismatch_res = {}, {}
    misaligned = []
    for pdb_file_pth in pdb_files_pth_list:
        pdb_fname = os.path.basename(pdb_file_pth)
        pdb_id = pdb_fname.split('.')[0][ :4]
        
        if pdb_id not in pdbs:
            print(f'The PDB {pdb_id} does not have corresponding item in the source csv file!')
            continue
        
        IUPHAR_name = pdb_list[pdb_list['PDB'] == pdb_id]['IUPHAR'].to_list()[0]
        # 将GN_A中所有列名里面的空格剔除再送回原处
        GN_A_all.rename(columns = {c: c.strip() for c in GN_A_all.columns.to_list()}, inplace = True)
        IUPHAR_list = GN_A_all.columns.to_list()[1: ]
        if IUPHAR_name not in IUPHAR_list:
            misaligned.append(pdb_id)
            print(f'The current PDB {pdb_id} has the IUPHAR name {IUPHAR_name}, but it does not exist in our data!')
            continue
        
        pdb_file_pth = os.path.join(pdb_files_dir, pdb_file_pth)
        dict_GN_to_pos, dict_mismatch_res = get_res_pos(pdb_file_pth, res_to_GN, pdb_list)
        if dict_GN_to_pos is None:
            misaligned.append(pdb_id)
            print(f'The PDB {pdb_id} should be dropped!')
            continue

        dict_all_res_pos[pdb_id] = dict_GN_to_pos
        dict_all_mismatch_res[pdb_id] = dict_mismatch_res
        
        if is_saved_json:
            with open('all_res_pos.json', 'w+') as f:
                f.write(json.dumps(dict_all_res_pos, ensure_ascii = False, indent = 2))
            
            with open('mismatch_res.json', 'w+') as f:
                f.write(json.dumps(dict_all_mismatch_res, ensure_ascii = False, indent = 2))
            with open("misalign.txt","w") as f:
                f.write("\n".join(misaligned))
    print(misaligned,len(misaligned))
if __name__ == '__main__':
    CSV_PTH = '/home/yangsy/projects/multi-state/GPCRdb_checked_v2.xlsx'
    PDB_FILES_DIR = '/home/yangsy/projects/multi-state/pdbs'

    get_all_res_pose(CSV_PTH, PDB_FILES_DIR)