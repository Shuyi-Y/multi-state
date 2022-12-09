#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import sys

# lddt torch version


def lddt_aa(pred_pose, real_pose, exist_mask, cutoff=15.0, peratom=False, mode="aa-aa"):
    '''
    :param pred_pose: num_res,14,3
    :param real_pose: num_res,14,3
    :param mask: num_res,14
    :return:
    '''
    pred_pose_reshape = pred_pose.reshape([-1, 3])
    real_pose_reshape = real_pose.reshape([-1, 3])
    exist_mask_reshape = exist_mask.reshape([-1]).to(torch.int)
    real_distmat = torch.sum((real_pose_reshape[None] - real_pose_reshape[:, None]) ** 2, dim=-1)
    pred_distmat = torch.sum((pred_pose_reshape[None] - pred_pose_reshape[:, None]) ** 2, dim=-1)
    exist_mask2d = exist_mask_reshape[None] * exist_mask_reshape[:, None] * (1 - torch.eye(exist_mask_reshape.shape[0]))
    sum_mask = (real_distmat < cutoff).to(torch.int) * exist_mask2d
    l1_disterror = torch.abs(pred_distmat - real_distmat)
    unnorm_score = 0.25 * ((l1_disterror < 0.5).to(torch.int) +
                           (l1_disterror < 1.0).to(torch.int) +
                           (l1_disterror < 2.0).to(torch.int) +
                           (l1_disterror < 4.0).to(torch.int))
    norm = 1 / (1E-10 + torch.sum(sum_mask, dim=0))
    per_atom_score = norm * (1E-10 + torch.sum(unnorm_score * sum_mask, dim=0))
    if mode == "aa-aa":
        if peratom:
            return per_atom_score
        else:
            return per_atom_score.mean()
    elif mode == "ca-aa":
        per_atom_score = per_atom_score.reshape([-1, 14])[:, 1]
        # The c-alpha atom is alwasy at index 1 of a residue
        if peratom:
            return per_atom_score
        else:
            return per_atom_score.mean()
    return per_atom_score



import io
from typing import Any, Mapping, Optional
from Bio.PDB import PDBParser
import numpy as np
import residue_constants

# parse atomcoord to [reslen,14,3]
            
def parse_pdb(pdb_fh: str, chain_id: Optional[str] = None):
    """Takes a PDB string and constructs a Protein object.
    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.
    Args:
    pdb_str: The contents of the pdb file
    chain_id: If chain_id is specified (e.g. A), then only that chain
      is parsed. Otherwise all chains are parsed.
    Returns:
    A new `Protein` parsed from the pdb contents.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
        f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.resname not in residue_constants.restype_3to1:
                continue
            if res.id[2] != ' ':
                raise ValueError(
                f'PDB contains an insertion code at chain {chain.id} and residue '
                f'index {res.id[1]}. These are not supported.')
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
              res_shortname, residue_constants.restype_num)
            pos = np.zeros((14, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                atom_types = residue_constants.restype_name_to_atom14_names[res.resname]
                atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
                pos[atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    atom_positions=np.array(atom_positions)
    return atom_positions
    
# if __name__ == "__main__":
#     refpdb = sys.argv[1]
#     refchain = sys.argv[2]
#     predpdb = sys.argv[3]
#     predchain = sys.argv[4]
#     r_p = torch.from_numpy(from_pdb(refpdb,refchain)) # reference pdb and chian name
#     p_p = torch.from_numpy(from_pdb(predpdb,predchain)) #predict pdb ...
#     e_m = torch.ones([r_p.shape[0],r_p.shape[1]])
#     lddt_score_peratom = lddt_aa(p_p, r_p, e_m, peratom=True, mode="aa-aa")
#     lddt_score_perres = torch.mean(lddt_score_peratom.reshape([-1,14]),dim = -1)
#     print(lddt_score_perres)

