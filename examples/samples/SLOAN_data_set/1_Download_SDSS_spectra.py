import os
import shutil
import pandas as pd
from pathlib import Path
from astroquery.sdss import SDSS
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_one(plate, mjd, fiber, out_dir, data_rel):

    # fname = os.path.join(out_dir, f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits")
    fname = out_dir/f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits"

    if not fname.is_file():
        try:
            res = SDSS.get_spectra(plate=plate, fiberID=fiber, mjd=mjd, data_release=data_rel, timeout=120)
            if not res:
                return (plate, mjd, fiber, "not_found")

            item = res[0]

            try:
                local_path = getattr(item, "filename", None) or getattr(item, "file", None)
                if callable(local_path):
                    local_path = local_path()
            except Exception:
                local_path = None

            if isinstance(item, dict) and "Local" in item:
                local_path = item["Local"]

            if local_path and os.path.exists(local_path):
                shutil.copy2(local_path, fname)
                return (plate, mjd, fiber, "ok")

            else:
                try:
                    from astropy.io import fits
                    if hasattr(item, "writeto"):
                        item.writeto(fname, overwrite=True)
                        return (plate, mjd, fiber, "ok")
                    elif hasattr(item, "to_hdu"):
                        hdu = item.to_hdu()
                        hdu.writeto(fname, overwrite=True)
                        return (plate, mjd, fiber, "ok")
                except Exception as e:
                    return (plate, mjd, fiber, f"save_error: {e}")

            return (plate, mjd, fiber, "unknown_format")

        except Exception as e:
            return (plate, mjd, fiber, f"error: {e}")
    else:
        return (plate, mjd, fiber, "ok")


def download_file_table(fname, save_folder, n_workers=10, data_rel=18):

    # Make folder if neccesary
    os.makedirs(save_folder, exist_ok=True)

    # Read your CSV; make column names case-insensitive
    df = pd.read_csv(fname, header=0, delimiter=',')
    cols = {c.lower(): c for c in df.columns}
    required = ["plate", "mjd", "fiberid"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(f"CSV must contain columns: {required} (case-insensitive). Missing: {missing}")

    # Parallel download
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(fetch_one, row.plate, row.mjd, row.fiberID, save_folder, data_rel)
                                            for row in df.itertuples(index=False)]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Report
    ok = [r for r in results if r[3] == "ok" or r[3] == "exists"]
    bad = [r for r in results if r[3] not in ("ok", "exists")]
    print(f"Done. Saved/existing: {len(ok)}, failures: {len(bad)}")
    if bad:
        print("Failures (plate, mjd, fiber, status):")
        for r in bad[:20]:
            print(r)

    return


def delete_all_files(folder_path, recursive=False):

    """
    Delete all files in a given folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder.
    recursive : bool, optional
        If True, delete files in subfolders as well. Default is False.
    """
    folder = Path(folder_path)

    if recursive:
        for file in folder.rglob("*"):
            if file.is_file():
                file.unlink()
    else:
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()

    return


# csv_sample = Path('SLOAN_data_set/Hbeta_SDSSdr18_low_SN.csv')
# output_folder = Path('/home/vital/Astrodata/SDSS_spectra/low_SN')
# download_file_table(csv_sample, output_folder)

csv_sample = Path('SLOAN_data_set/Hbeta_SDSSdr18_high_SN.csv')
output_folder = Path('/home/vital/Astrodata/SDSS_spectra/high_SN')
# delete_all_files(output_folder)
download_file_table(csv_sample, output_folder)








