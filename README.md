# FATT

FATT is a renaming of  https://github.com/abstract-machine-learning/meta-silvae to ensure a blind review process. This repository allows to replicate experimental data.

## Experiments

Experimental data can be replicated by cloning this repository and running `make`:

```bash
git clone https://github.com/fatt21/fatt.git
cd fatt
make
```

this will download and pre-process the necessary dataset, will train decision tree models and verify their fairness. Intermediate output files will be written under `output`, while actual tables/figures will be generated under `results`.

**Note**: code in this repository runs CPU-intense machine learning training and verification tasks, which may be not suitable for home computers. Most experiment are repeated a number of times (usually 1000) to ensure statistical relevance, resulting in high running time and overall computational costs.

## Requirements

* Python3
* Any C99 compiler
* Ghostscript
* Gnuplot