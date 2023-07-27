from rdkit import Chem
from rdkit.Chem import rdChemReactions
import pandas as pd



def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]

def generate_buchwald_hartwig_rxns(df):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]' 
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template) 
    products = []
    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol) 
        rxn_products = rxn.RunReactants(reacts)
        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    df['product'] = products
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive
        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append(f"{reactants}>>{row['product']}")
    return rxns

if __name__ == '__main__':
    
    expected_rxns = ['Clc1ccccn1.Cc1ccc(N)cc1.O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F.COc1ccc(OC)c(P([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)[C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C.CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C.Cc1cc(C)on1>>Cc1ccc(Nc2ccccn2)cc1',
     'Brc1ccccn1.Cc1ccc(N)cc1.O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F.COc1ccc(OC)c(P([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)[C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C.CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C.COC(=O)c1ccno1>>Cc1ccc(Nc2ccccn2)cc1',
     'CCc1ccc(I)cc1.Cc1ccc(N)cc1.O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F.CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C.COC(=O)c1ccno1>>CCc1ccc(Nc2ccc(C)cc2)cc1',
     'FC(F)(F)c1ccc(Cl)cc1.Cc1ccc(N)cc1.O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F.COc1ccc(OC)c(P(C(C)(C)C)C(C)(C)C)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C.CN1CCCN2CCCN=C12.CCOC(=O)c1cnoc1>>Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1',
     'COc1ccc(Cl)cc1.Cc1ccc(N)cc1.O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F.COc1ccc(OC)c(P([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)[C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C.CN1CCCN2CCCN=C12.Cc1cc(C)on1>>COc1ccc(Nc2ccc(C)cc2)cc1']
    
    test_df = pd.DataFrame({"Ligand":{"0":"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC","1":"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC","2":"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2","3":"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC","4":"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC"},"Additive":{"0":"CC1=CC(C)=NO1","1":"O=C(OC)C1=CC=NO1","2":"O=C(OC)C1=CC=NO1","3":"CCOC(C1=CON=C1)=O","4":"CC1=CC(C)=NO1"},"Base":{"0":"CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC","1":"CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC","2":"CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC","3":"CN1CCCN2C1=NCCC2","4":"CN1CCCN2C1=NCCC2"},"Aryl halide":{"0":"ClC1=NC=CC=C1","1":"BrC1=NC=CC=C1","2":"IC1=CC=C(CC)C=C1","3":"ClC1=CC=C(C(F)(F)F)C=C1","4":"ClC1=CC=C(OC)C=C1"},"Output":{"0":70.41045785,"1":11.06445724,"2":10.22354965,"3":20.0833829,"4":0.492662711}})
    
    converted_rxns = generate_buchwald_hartwig_rxns(test_df) 
    
    for rxn, expected in zip(converted_rxns, expected_rxns):
        assert rxn == expected