# Pyxel Data

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/esa%2Fpyxel-data/HEAD)
[![Jupyter Book](https://jupyterbook.org/badge.svg)](https://esa.gitlab.io/pyxel-data)


This repository contains example notebooks for Pyxel and data that would clutter the main Pyxel repository. 
This repository can be downloaded and examples can be run locally.

They are also available on Binder, without prior Pyxel installation, by clicking on the badge above.

## Quickstart Setup

The best way to get started and learn Pyxel are the [Tutorials and Examples](https://esa.gitlab.io/pyxel/doc/stable/tutorials/overview.html#quickstart-setup)

For convenience we provide a pre-defined conda environment file,
so you can get additional useful packages together with Pyxel in a virtual isolated environment.

First install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then just execute the following
commands in the terminal:

```bash
curl -O https://esa.gitlab.io/pyxel/doc/latest/pyxel-2.1.1-environment.yaml
conda env create -f pyxel-2.1.1-environment.yaml
```

Once the conda environment has been created you can active it using:

```bash
conda activate pyxel-2.1.1
```

You can now proceed to download the Pyxel tutorial notebooks.
The total size to download is ~200 MB.

Select the location where you want to install the tutorials and datasets and
proceed with the following command to download them in folder ``pyxel-examples``:

```bash
pyxel download-examples
```

Finally start a notebook server by executing:

```bash
cd pyxel-examples
jupyter lab
```

Now, you can skip the installation guide [install](https://esa.gitlab.io/pyxel/doc/stable/tutorials/install.html) and 
go directly to the tutorials and explore the examples in [examples](https://esa.gitlab.io/pyxel/doc/stable/tutorials/examples.html) 
to learn how to use Pyxel.


## Clone Pyxel Data

Instructions to clone Pyxel Data:

```bash
git clone https://gitlab.com/esa/pyxel-data.git
cd pyxel-data
jupyter lab
```

## Download Pyxel Data

The preferred way to get Pyxel data is to download it.

### Download manually

You can download Pyxel Data directly as a zip file by clicking the '[download](https://gitlab.com/esa/pyxel-data/-/archive/master/pyxel-data-master.zip)' link next to the 'clone' link.

### Download via Pyxel

If Pyxel is already installed, you can download Pyxel Data with the following command:

```bash
pyxel download-examples
cd pyxel-examples
jupyter lab
```

## Links

[![docs](https://esa.gitlab.io/pyxel/documentation.svg)](https://esa.gitlab.io/pyxel/doc)
[![gitter](https://badges.gitter.im/pyxel-framework/community.svg)](https://gitter.im/pyxel-framework/community)
[![Google Group](https://img.shields.io/badge/Google%20Group-Pyxel%20Detector%20Framework-blue.svg)](https://groups.google.com/g/pyxel-detector-framework)
[![doi](https://zenodo.org/badge/DOI/10.1117/1.JATIS.8.4.048002.svg)](https://doi.org/10.1117/1.JATIS.8.4.048002)

## Build the documentation

### Install Jupyter Book

```bash
conda install -c conda-forge jupyter-book ipympl pyxel-sim
```

### Re-build the documentation

```bash
jupyter-book clean --all .
jupyter-book build .
```

or 

```bash
pip install -e tox
tox -e build_latest_dev
```
