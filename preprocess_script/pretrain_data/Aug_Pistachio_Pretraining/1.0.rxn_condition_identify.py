import pandas as pd 
import json
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm
from collections import namedtuple, defaultdict
from rdkit.Chem.SaltRemover import SaltRemover, InputFormat
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')




class MolRemover(SaltRemover):
    def __init__(self, defnFilename=None, defnData=None, defnFormat=InputFormat.SMARTS):
        super().__init__(defnFilename, defnData, defnFormat)

    def _StripMol(self, mol, dontRemoveEverything=False, onlyFrags=False):

        def _applyPattern(m, salt, notEverything, onlyFrags=onlyFrags):
            nAts = m.GetNumAtoms()
            if not nAts:
                return m
            res = m

            t = Chem.DeleteSubstructs(res, salt, onlyFrags)
            if not t or (notEverything and t.GetNumAtoms() == 0):
                return res
            res = t
            while res.GetNumAtoms() and nAts > res.GetNumAtoms():
                nAts = res.GetNumAtoms()
                t = Chem.DeleteSubstructs(res, salt, True)
                if notEverything and t.GetNumAtoms() == 0:
                    break
                res = t
            return res

        org_mol = mol
        StrippedMol = namedtuple('StrippedMol', ['mol', 'deleted'])
        deleted = []
        if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
            return StrippedMol(mol, deleted)
        modified = False
        natoms = mol.GetNumAtoms()
        for salt in self.salts:
            mol = _applyPattern(mol, salt, dontRemoveEverything, onlyFrags)
            if natoms != mol.GetNumAtoms():
                natoms = mol.GetNumAtoms()
                modified = True
                deleted.append(salt)
                if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
                    break
        if modified and mol.GetNumAtoms() > 0:
            try: 
                Chem.SanitizeMol(mol)
                return StrippedMol(mol, deleted)
            except:
                return StrippedMol(org_mol, '')
        return StrippedMol(mol, deleted)

    def StripMolWithDeleted(self, mol, dontRemoveEverything=False, onlyFrags=False):
        return self._StripMol(mol, dontRemoveEverything, onlyFrags=onlyFrags)


if __name__ == '__main__':
    df = pd.read_csv('../../dataset/pretrain_data/pistachio_pretraining.csv', encoding='utf-8')
    with open('../../dataset/pretrain_data/condition_identify.json', encoding='utf-8') as json_file:
        condition_dict = json.load(json_file)
    cat_ls = condition_dict['catalyst']
    sol_ls = condition_dict['solvent']
    rea_ls = condition_dict['reagent']

    catalyst = []
    solvent = []
    reagent = []
    need_match = []
    maybe_salt = []
    for final_cdt in tqdm(df['final_can_regent'].tolist(), total=len(df)):
        if pd.isna(final_cdt):
            catalyst.append('')
            solvent.append('')
            reagent.append('')
            need_match.append('')
            maybe_salt.append('')
        else:
            if "+" in final_cdt or "-" in final_cdt:
                catalyst.append('')
                solvent.append('')
                reagent.append('')
                need_match.append('')
                maybe_salt.append(final_cdt)
            else:
                maybe_salt.append('')
                cat_ = []
                rea_ = []                
                sol_ = []
                matchs = []
                cdts = final_cdt.split('.')
                for cdt in cdts:
                    if cdt in cat_ls:
                        cat_.append(cdt)
                    elif cdt in rea_ls:
                        rea_.append(cdt)
                    elif cdt in sol_ls:
                        sol_.append(cdt)
                    else:
                        matchs.append(cdt)
                if cat_:
                    cat_ = ';'.join(cat_)
                    catalyst.append(cat_)
                else:
                    catalyst.append('')
                
                if rea_:
                    rea_ = ';'.join(rea_)
                    reagent.append(rea_)
                else:
                    reagent.append('')   
                if sol_:
                    sol_ = ';'.join(sol_)
                    solvent.append(sol_)
                else:
                    solvent.append('') 
                if matchs:
                    matchs = ';'.join(matchs)
                    need_match.append(matchs)
                else:
                    need_match.append('') 
    df['catalyst'] = catalyst
    df['reagent'] = reagent
    df['solvent'] = solvent
    df['maybe_salt'] = maybe_salt    
    df['need_match'] = need_match  

    reagent_1 = []
    remain_1 = []
    for (rgt, sl) in tqdm(zip(df['reagent'].tolist(), df['maybe_salt'].tolist()), total=len(df)):
        if sl == '':
            reagent_1.append(rgt)
            remain_1.append('')        
        else: 
            sl_ls  = sl.split('.')
            if len(sl_ls) == 1: 
                reagent_1.append(rgt)  
                remain_1.append(sl)
            else:
                rgt_ls = rgt.split(';')
                sl_mol = Chem.MolFromSmiles(sl)
                remover = MolRemover(defnFilename='../../dataset/pretrain_data/reagent_Ionic_compound.txt')
                strippedmol, deletedmols = remover.StripMolWithDeleted(
                sl_mol, dontRemoveEverything=False, onlyFrags=False)      
                remain_str = Chem.MolToSmiles(strippedmol)
                # remain_1.append(remain_str)
                if deletedmols:
                    delete_str = [Chem.MolToSmiles(x) for x in deletedmols]
                    if len(sl_ls) == len(remain_str.split('.')) + len(delete_str):
                        rgt_ls.extend(delete_str)
                        rgt_ls = ';'.join(rgt_ls)
                        reagent_1.append(rgt_ls)
                        remain_1.append(remain_str)
                    else:
                        reagent_1.append(rgt)
                        remain_1.append(sl)
                else:
                    reagent_1.append(rgt)
                    # remain_1.append(remain_str)
                    remain_1.append(sl)
    
    assert len(reagent_1) == len(remain_1)
    df['reagent'] = reagent_1
    df['remain_1'] = remain_1

    c_ = []
    r_ = []
    s_ = []
    need_ = []
    for (c, r, s, need, re_) in tqdm(zip(df['catalyst'].tolist(), df['reagent'].tolist(), df['solvent'].tolist(), df['need_match'].tolist(), df['remain_1'].tolist()), total=len(df)):
        if re_ == '':
            c_.append(c)
            r_.append(r)
            s_.append(s)
            need_.append(need)
        else:
            cs = c.split(';')
            rs = r.split(';')
            ss = s.split(';')
            needs = need.split(';')
            re_ls = re_.split('.')
            for molecule in re_ls:
                if molecule in cat_ls:
                    cs.append(molecule)
                elif molecule in rea_ls:
                    rs.append(molecule)
                elif molecule in sol_ls:
                    ss.append(molecule)
                else:
                    needs.append(molecule)
            cs = ';'.join(cs)
            c_.append(cs)
            rs = ';'.join(rs)
            r_.append(rs)
            ss = ';'.join(ss)
            s_.append(ss)
            needs = ';'.join(needs)
            need_.append(needs)
    assert len(c_) == len(r_)
    assert len(c_) == len(s_)
    assert len(c_) == len(need_)
    df['catalyst'] = c_
    df['reagent'] = r_
    df['solvent'] = s_
    df['need_match'] = need_

    cs_ = []
    rs_ = []
    ss_ = []
    unknown = []
    for (c_1, r_1, s_1, need_1) in tqdm(zip(df['catalyst'].tolist(), df['reagent'].tolist(), df['solvent'].tolist(), df['need_match'].tolist()), total=len(df)): 
        if need_1 == '':
            cs_.append(c_1)
            rs_.append(r_1)
            ss_.append(s_1)
            unknown.append('')
        else:
            c_ls = c_1.split(';')
            r_ls = r_1.split(';')
            s_ls = s_1.split(';')
            unknown_ls = []
            need_ls = need_1.split(';')
            for need_ in need_ls:
                if need_ in cat_ls:
                    c_ls.append(need_)
                elif need_ in rea_ls:
                    r_ls.append(need_)
                elif need_ in sol_ls:
                    s_ls.append(need_)
                else:
                    unknown_ls.append(need_)
            c_ls = ';'.join(c_ls)
            cs_.append(c_ls)
            r_ls = ';'.join(r_ls)
            rs_.append(r_ls)
            s_ls = ';'.join(s_ls)
            ss_.append(s_ls)
            unknown_ls = ';'.join(unknown_ls)
            unknown.append(unknown_ls)        
    assert len(cs_) == len(rs_)
    assert len(cs_) == len(ss_)
    assert len(cs_) == len(unknown)
    df['catalyst'] = cs_
    df['reagent'] = rs_
    df['solvent'] = ss_
    df['unknown'] = unknown

    id_list = [i for i in range(len(df))]
    df['_ID'] = id_list
    df.to_csv('../../dataset/pretrain_data/pistachio_reaction_condition_identify.csv', encoding='utf-8', index=False)  
                                               









    



    
            

            
