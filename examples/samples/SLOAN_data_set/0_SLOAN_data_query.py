from astroquery.sdss import SDSS
import pandas as pd
import lime
import matplotlib.pyplot as plt


def run_query(query_text, fname):

    # Run query on DR18
    result = SDSS.query_sql(query_text, data_release=18)

    # Convert to DataFrame
    if result is not None:
        df = result.to_pandas()
        df.to_csv(fname, index=False)
        lime.save_frame(fname.replace('csv', 'txt'), df)
        print(f"Saved {len(df)} rows to {fname}")
    else:
        print("No results returned.")

    return


def plot_variable_histogram(var_name, sql_df, fname=None):

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sql_df[var_name], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel(var_name)
    ax.set_ylabel('Count')
    ax.set_title(f'{var_name} histogram')
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

    return


# High S/N sample
fname = './Hbeta_SDSSdr18_high_SN.csv'
sql_query_high = """
SELECT TOP 10000
    s.plate, s.mjd, s.fiberID, s.ra, s.dec, s.z,
    e.bpt,
    (e.Flux_Hb_4861 / NULLIF(e.Flux_Hb_4861_Err, 0)) AS Hb_SN,
    (e.Flux_OIII_4958 / NULLIF(e.Flux_OIII_4958_Err, 0)) AS O3_4959A_SN,
    (e.Flux_OIII_5006 / NULLIF(e.Flux_OIII_5006_Err, 0)) AS O3_5007A_SN,
    e.Amplitude_Hb_4861, e.Amplitude_Hb_4861_Err,
    e.Amplitude_OIII_5006, e.Amplitude_OIII_5006_Err,
    e.Sigma_Hb_4861, e.Sigma_Hb_4861_Err,
    e.Sigma_OIII_5006, e.Sigma_OIII_5006_Err,
    e.Flux_Hb_4861, e.Flux_Hb_4861_Err,
    e.Flux_OIII_4958, e.Flux_OIII_4958_Err,
    e.Flux_OIII_5006, e.Flux_OIII_5006_Err,
    e.Flux_Cont_Hb_4861, e.Flux_Cont_Hb_4861_Err,
    e.Flux_Cont_OIII_5006, e.Flux_Cont_OIII_5006_Err, 
    e.EW_Hb_4861, e.EW_Hb_4861_Err,
    e.EW_OIII_5006, e.EW_OIII_5006_Err,                    
    e.Fit_Warning_Hb_4861,
    e.Fit_Warning_OIII_4958,
    e.Fit_Warning_OIII_5006
FROM SpecObjAll AS s
JOIN emissionLinesPort AS e
    ON s.specObjID = e.specObjID
WHERE s.sciencePrimary = 1
  AND s.class = 'GALAXY'
  AND s.zWarning = 0
  AND e.bpt = 'Star Forming'
  AND e.Flux_Hb_4861 IS NOT NULL
  AND e.Flux_OIII_4958 IS NOT NULL
  AND e.Flux_OIII_5006 IS NOT NULL
  AND e.Flux_Hb_4861_Err > 0
  AND e.Flux_OIII_4958_Err > 0
  AND e.Flux_OIII_5006_Err > 0
  AND e.Flux_Cont_Hb_4861_Err > 0
  AND e.EW_OIII_5006_Err BETWEEN 0 AND 100
  AND (e.Flux_OIII_5006 / e.Flux_OIII_5006_Err) BETWEEN 10 AND 100
  AND (e.Flux_OIII_5006 / e.Flux_OIII_5006_Err) BETWEEN 10 AND 50
  AND e.Fit_Warning_Hb_4861 = 0
  AND e.Fit_Warning_OIII_4958 = 0
  AND e.Fit_Warning_OIII_5006 = 0
"""

# Check the files
run_query(sql_query_high, fname)

# Load the sample
df = pd.read_csv(fname, header=0, delimiter=',')

var = 'O3_5007A_SN'
plot_variable_histogram(var, df, f'./{var}_high_SN_histogram.png')

### -------------------------------------------------------------------------------

# High S/N sample
fname = './Hbeta_SDSSdr18_low_SN.csv'
sql_query_high = """
SELECT TOP 10000
    s.plate, s.mjd, s.fiberID, s.ra, s.dec, s.z,
    e.bpt,
    (e.Flux_Hb_4861 / NULLIF(e.Flux_Hb_4861_Err, 0)) AS Hb_SN,
    (e.Flux_OIII_4958 / NULLIF(e.Flux_OIII_4958_Err, 0)) AS O3_4959A_SN,
    (e.Flux_OIII_5006 / NULLIF(e.Flux_OIII_5006_Err, 0)) AS O3_5007A_SN,
    e.Amplitude_Hb_4861, e.Amplitude_Hb_4861_Err,
    e.Amplitude_OIII_5006, e.Amplitude_OIII_5006_Err,
    e.Sigma_Hb_4861, e.Sigma_Hb_4861_Err,
    e.Sigma_OIII_5006, e.Sigma_OIII_5006_Err,
    e.Flux_Hb_4861, e.Flux_Hb_4861_Err,
    e.Flux_OIII_4958, e.Flux_OIII_4958_Err,
    e.Flux_OIII_5006, e.Flux_OIII_5006_Err,
    e.Flux_Cont_Hb_4861, e.Flux_Cont_Hb_4861_Err,
    e.Flux_Cont_OIII_5006, e.Flux_Cont_OIII_5006_Err,
    e.EW_Hb_4861, e.EW_Hb_4861_Err,
    e.EW_OIII_5006, e.EW_OIII_5006_Err,
    e.Fit_Warning_Hb_4861,
    e.Fit_Warning_OIII_4958,
    e.Fit_Warning_OIII_5006
FROM SpecObjAll AS s
JOIN emissionLinesPort AS e
    ON s.specObjID = e.specObjID
WHERE s.sciencePrimary = 1
  AND s.class = 'GALAXY'
  AND s.zWarning = 0
  AND e.bpt = 'Star Forming'
  AND e.Flux_Hb_4861 IS NOT NULL
  AND e.Flux_OIII_4958 IS NOT NULL
  AND e.Flux_OIII_5006 IS NOT NULL
  AND e.Flux_Hb_4861_Err > 0
  AND e.Flux_OIII_4958_Err > 0
  AND e.Flux_OIII_5006_Err > 0
  AND e.Flux_Cont_Hb_4861_Err > 0
  AND e.EW_OIII_5006_Err BETWEEN 0 AND 100
  AND (e.Flux_OIII_5006 / e.Flux_OIII_5006_Err) BETWEEN 2 AND 100
  AND (e.Flux_OIII_5006 / e.Flux_OIII_5006_Err) BETWEEN 2 AND 10
  AND e.Fit_Warning_Hb_4861 = 0
  AND e.Fit_Warning_OIII_4958 = 0
  AND e.Fit_Warning_OIII_5006 = 0
"""

# Check the files
run_query(sql_query_high, fname)

# Load the sample
df = pd.read_csv(fname, header=0, delimiter=',')

var = 'O3_5007A_SN'
plot_variable_histogram(var, df, f'./{var}_low_SN_histogram.png')
