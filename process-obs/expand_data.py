import os, zipfile, sys
from tqdm import tqdm

dir = os.path.join("D:\\argonne_data", "IGRA_v2.2_data-por_s19050404_e20230606_c20230606")
zipfiles = [file for file in os.listdir(dir) if '.zip' in file]
print('Num. zip files: %d' % len(zipfiles))
for file in tqdm(zipfiles):
    if not os.path.exists(os.path.join(dir, '.'.join(file.split('.')[:-1]))):
        with zipfile.ZipFile(os.path.join(dir, file), 'r') as zip_ref:
            zip_ref.extractall(dir)
        os.remove(os.path.join(dir, file))
txtfiles = [file for file in os.listdir(dir) if '.txt' in file]
print('Num .txt files: %d' % len(txtfiles))

