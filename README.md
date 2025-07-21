# humancompatible.repair
An open-source toolkit for post-hoc verification of fairness and repair thereof. If you like the code, please cite:

```
@misc{zhou2024groupblindoptimaltransportgroup,
      title={Group-blind optimal transport to group parity and its constrained variants}, 
      author={Quan Zhou and Jakub Marecek},
      year={2024},
      eprint={2310.11407},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.11407}, 
}
```

## Installation

### (Optional) create a fresh environment
```bash
python -m venv .venv
# ── Activate it ─────────────────────────────────────────────
# Linux / macOS
source .venv/bin/activate
# Windows – cmd.exe
.venv\Scripts\activate.bat
# Windows – PowerShell
.venv\Scripts\Activate.ps1
```

### Install the package

> **Package not on PyPI yet?**  
> Until we complete the PyPI release you can install the latest snapshot
> straight from GitHub in one line:

```bash
python -m pip install git+https://github.com/humancompatible/repair.git
```

If you prefer an editable (developer) install:

```bash
git clone https://github.com/humancompatible/repair.git
cd repair
python -m pip install -r requirements.txt
python -m pip install -e .
```

**In case it throws an error, you may probably need to create a folder `plots`, or `data` in the root folder of the project.  
So the project knows where to store the reports.**


