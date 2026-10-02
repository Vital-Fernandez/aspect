(installation-doc)=
# Installation

$\mathrm{ASPECT}$ can be installed via [pip](https://pypi.org/project/aspect-stable/):

```bash
pip install aspect-stable
```

:::{tip}
$\mathrm{ASPECT}$ is also included in the [$\mathrm{LiMe}$](https://lime-stable.readthedocs.io/) installation, which
provides the tools to run it directly on your spectra:

```bash
pip install lime-stable
```
:::

## Core Dependencies

The following packages are essential for $\mathrm{ASPECT}$'s operation:

- [Numpy](https://numpy.org/install/) – Array operations  
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) – Table management and reading tabulated files  
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) – Default plotting library  
- [scikit-learn](https://scikit-learn.org/stable/install.html) – Machine learning models  
- [joblib](https://joblib.readthedocs.io/en/stable/installing.html) – Loading the trained model files  

## Updating ASPECT

To upgrade to the latest version:

```bash
pip install aspect-stable --upgrade
```

## Uninstalling or Changing Versions

To uninstall $\mathrm{ASPECT}$:

```bash
pip uninstall aspect-stable
```

To install a specific version:

```bash
pip install aspect-stable==0.9
```

## For Developers

The following command installs all dependencies required to compile the documentation and run tests:

```bash
pip install aspect-stable[docs,tests]
```

:::{warning}
Depending on the terminal shell language, you may need to use quotes in the command with the dependencies:
```bash
pip install "aspect-stable[docs,tests]"
```
:::

**ASPECT v0.9** requires **Python 3.12** or later. See the [pyproject.toml](https://github.com/Vital-Fernandez/aspect/blob/main/pyproject.toml)
on GitHub for the most up-to-date build and dependency information.