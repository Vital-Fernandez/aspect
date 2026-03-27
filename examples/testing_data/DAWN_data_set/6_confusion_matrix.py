import lime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Read sample database
sample_fname = './aspect_DAWN_prism_v4_measurements.csv'
sample_df = lime.load_frame(sample_fname)
sample_df = sample_df.iloc[502:].reset_index(drop=True)

my_palette = {"neither": "#fbb4ae", "zxor_only": "#b3cde3",
              "zkey_only": "#decbe4", "both": "#ccebc5"}

# Create a reusable colormap for heatmaps
custom_cmap = ListedColormap([my_palette["neither"], my_palette["zxor_only"],
                              my_palette["zkey_only"], my_palette["both"]])

# None NaN columns
mask = (sample_df["z"].notna() & sample_df["zxor"].notna() & sample_df["zkey"].notna())
df_clean = sample_df.loc[mask]

# Assuming your dataframe is named 'df'
threshold = 0.05

df_clean['match_key'] = (abs(df_clean['zkey'] - df_clean['z']) / df_clean['z']) <= threshold
df_clean['match_xor'] = (abs(df_clean['zxor'] - df_clean['z']) / df_clean['z']) <= threshold

# Create the cross-tabulation (the actual confusion matrix data)
ct = pd.crosstab(df_clean['match_key'], df_clean['match_xor'])

# Ensure the matrix is 2x2 even if a category has 0 counts
ct = ct.reindex(index=[False, True], columns=[False, True], fill_value=0)

# --- 3. PREPARE ANNOTATIONS & COLOR MAPPING ---
cell_labels = np.array([[f"Neither Matches\n{ct.iloc[0,0]}", f"Only zxor Matches\n{ct.iloc[0,1]}"],
                        [f"Only zkey Matches\n{ct.iloc[1,0]}", f"Both Match\n{ct.iloc[1,1]}"]])

# This dummy matrix (0-3) ensures each cell maps to a specific color in our palette
color_mapping = np.array([[0, 1], [2, 3]])

# --- 4. PLOTTING ---
plt.figure(figsize=(10, 8))

sns.heatmap(color_mapping, annot=cell_labels, fmt="", cmap=custom_cmap, cbar=False,
            linecolor='white', xticklabels=["zxor: No Match", "zxor: Match"], yticklabels=["zkey: No Match", "zkey: Match"])

plt.title(f"Methodology Comparison (Tolerance: {threshold*100}%)", fontsize=16, pad=20)
plt.xlabel("zxor Prediction Status", fontsize=12)
plt.ylabel("zkey Prediction Status", fontsize=12)

# Optional: Adjust layout to prevent clipping
plt.tight_layout()
plt.show()



'''
Cell LocationLabelingWhat it means for your dataBottom-RightMatches / MatchesThe "Gold Standard" group. Both zkey and zxor are within $5\%$ of the true value $Z$. Both methodologies are performing well on these specific objects.Top-LeftDoesn't Match / Doesn't MatchThe "Hard" group. Neither methodology could predict $Z$ accurately. This might suggest these specific data points are outliers or inherently harder to predict.Bottom-LeftMatches (zkey) / Doesn't Match (zxor)The "zkey wins" group. Only zkey was within $5\%$. This shows you exactly where zkey is more robust or accurate than zxor.Top-RightDoesn't Match (zkey) / Matches (zxor)The "zxor wins" group. Only zxor was within $5\%$. This highlights the cases where zxor is superior to zkey.

Cell Location,Labeling,What it means for your data
Bottom-Right,Matches / Matches,"The ""Gold Standard"" group. Both zkey and zxor are within 5% of the true value Z. Both methodologies are performing well on these specific objects."
Top-Left,Doesn't Match / Doesn't Match,"The ""Hard"" group. Neither methodology could predict Z accurately. This might suggest these specific data points are outliers or inherently harder to predict."
Bottom-Left,Matches (zkey) / Doesn't Match (zxor),"The ""zkey wins"" group. Only zkey was within 5%. This shows you exactly where zkey is more robust or accurate than zxor."
Top-Right,Doesn't Match (zkey) / Matches (zxor),"The ""zxor wins"" group. Only zxor was within 5%. This highlights the cases where zxor is superior to zkey."

'''


# 'title': f'Dawn galaxy redshift comparison ({np.sum(idcs)}/{sample_df.index.size})',

# # Trendlines
# axes[0].plot([0, z_limit], [0, z_limit], 'r--', label='x = y')
# axes[1].plot([0, z_limit], [0, z_limit], 'r--', label='x = y')
#
# # Data
# x, y = sample_df.loc[idcs_zor].zkey.to_numpy(), sample_df.loc[idcs_zor].z.to_numpy()
# axes[0].scatter(x, y, alpha=0.2, edgecolors='none')
#
# x, y = sample_df.loc[idcs_zkey].zkey.to_numpy(), sample_df.loc[idcs_zkey].z.to_numpy()
# axes[1].scatter(x, y, alpha=0.2, edgecolors='none')
#
# # Plot format
# axes[0].grid(True, which='both', linewidth=0.5, alpha=0.7)
# axes[1].grid(True, which='both', linewidth=0.5, alpha=0.7)
# axes[0].set_xlim(0, z_limit)
# axes[1].set_xlim(0, z_limit)
# axes[0].set_ylim(0, z_limit)
# axes[1].set_ylim(0, z_limit)
#
# # Plot wording
# axes[0].set(**{'xlabel': f'Aspect redshift (Pixel count)', 'ylabel': f'True redshift (DAWN)'})
# axes[1].set(**{'xlabel': f'Aspect redshift (Flux count)'})


# I have a dataframe where the column 'z' is the true value and columns "zxor" and "zkey" are the predictions from two different techniques. Can you give me the python code to plot the confusion matrix showing the relation between true values and the predicitons