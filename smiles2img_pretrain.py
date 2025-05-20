import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png

        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================

    '''
    mol = Chem.MolFromSmiles(smis[0])
    # mol = Chem.AddHs(mol)  # 添加氢原子（如果需要）

    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
    img.save(path)


def main():
    '''
    demo of raw_file_path:
        index,k_100,k_1000,k_10000,smiles
        1,97,861,4925,CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1
        2,37,524,4175,CC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1
        3,77,636,6543,COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC
        ...
    :return:
    '''
    parser = argparse.ArgumentParser(description='Pretraining Data Generation for ImageMol')
    parser.add_argument('--dataroot', type=str, default="./pretraining/", help='data root')
    parser.add_argument('--dataset', type=str, default="pubchem-5m", help='dataset name, e.g. data')
    args = parser.parse_args()

    raw_file_path = os.path.join(args.dataroot, args.dataset, "{}.csv".format(args.dataset))
    img_save_root = os.path.join(args.dataroot, args.dataset, "225")
    # csv_save_path = os.path.join(args.dataroot, args.dataset, "{}_for_pretrain.csv".format(args.dataset))
    # error_save_path = os.path.join(args.dataroot, args.dataset, "error_smiles.csv")

    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path)
    smileslist = df.values
    i = 2

    for smiles in smileslist:
        # if i <= 1:
        #     i = i + 1
        #     continue

        print('i=', i)
        filename = "{}.png".format(i)  #+smiles[0]+".png"

        with open('indices.txt', 'a') as f:  #分子表达式通过.png名查看indices.txt
            # f.write('\n')  # 添加空行分隔新旧数据
            f.write(str(i))
            f.write(str(smiles[0]) +'\n')


        img_save_path = os.path.join(img_save_root, filename)
        loadSmilesAndSave(smiles, img_save_path)
        i = i+1


if __name__ == '__main__':
    main()
