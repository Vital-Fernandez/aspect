import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime
import requests

from pathlib import Path

# Read dawn table
fname = './nirspec_graded_v3.csv'
df = pd.read_csv(fname, header=0)
df = df.drop(columns='FITS')
for column in ['References', 'comment']:
    column_arr = df[column]
    df = df.drop(columns=column)
    df[column] = column_arr

# Index the selection
idcs_match = np.isclose(df.zfit.to_numpy(), df.z.to_numpy(), rtol=0.05)
idcs = (df.grade == 3) & idcs_match & (df.Ha > 20) & (df.z > 0.5) & (df.z < 6.0)
print(f'Total entries: {df.loc[idcs].shape[0]}')
df_selection = df.loc[idcs]
df_selection = df_selection.reset_index(drop=True)

df_selection.insert(13, 'zkey', np.nan)
df_selection.insert(14, 'zxor', np.nan)

fname = './aspect_DAWN_prism_v3_selection.csv'
lime.save_frame(fname, df_selection)

# Histogram with the redshifts
fig, ax = plt.subplots(figsize=(8, 5), dpi=350)
df_selection['z'].hist(bins=10, ax=ax, color='C0', edgecolor='black')
ax.set_xlabel('Redshift $z$')
ax.set_ylabel('Number of objects')
ax.set_title(f'JWST DAWN spectra selection (N = {df_selection.index.size}/{df.index.size})')
plt.tight_layout()
plt.savefig('./DAWN_prism_v3_selection.png')

# Download the spectra
BASE_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'
output_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = df_selection.root.to_numpy(), df_selection.file.to_numpy()
urls = [f"{BASE_URL}{root}/{file}" for root, file in zip(root_arr, file_arr)]

# Make root folder if necessary
for root in np.unique(root_arr):
    subfolder = output_dir/root
    subfolder.mkdir(parents=True, exist_ok=True)

for i, url in enumerate(urls):
    print(f"Downloading {file_arr[i]} ...")
    saving_path = output_dir / root_arr[i] / file_arr[i]

    if not saving_path.is_file():
        response = requests.get(url, stream=True)
        if response.status_code == 200 :
            with open(saving_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" - Saved to {saving_path}")
        else:
            print(f" - Failed to download {url} (status code: {response.status_code})")
    else:
        print(f" - File already exists")
