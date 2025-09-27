
---

# Inverse Projection Testing

This repository contains code and notebooks for the **Inverse Projection Framework**, which supports **analysis, visualization, and experiments**. It includes sample datasets, generated results, and a primary Jupyter notebook to run the main projection analysis.

---

## Repository Structure

| Folder / File                    | Description                                                        |
| -------------------------------- | ------------------------------------------------------------------ |
| `datasets/`                      | Sample or required input data files to run the analysis.           |
| `results/`                       | Generated outputs, plots, and visualizations are stored here.      |
| `main_projection_analysis.ipynb` | Main Jupyter notebook executing the framework’s analysis pipeline. |
| `pyproject.toml`                 | Poetry configuration file listing dependencies.                    |
| `poetry.lock`                    | Locked versions of dependencies for reproducibility.               |

---

## Prerequisites

Before running the project, ensure you have:

* **Python 3.12**
* **Poetry** (for dependency management)
* **Jupyter Notebook** or **Jupyter Lab**

---

## Installing Poetry

If Poetry is not installed on your system, follow these instructions:

### On Linux/macOS:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### On Windows (PowerShell):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

After installation, verify it by running:

```bash
poetry --version
```

---

## Setting Up the Project

1. Clone this repository:

```bash
git clone https://github.com/Waqar351/inverse_projection_testing.git
cd inverse_projection_testing
```

2. Install project dependencies using Poetry:

```bash
poetry install --no-root
```

This will automatically create a virtual environment and install all required packages.

---

## Running the Notebook

1. Activate the Poetry virtual environment:

```bash
poetry shell
```

2. Launch Jupyter Notebook or Jupyter Lab from the repository directory:

```bash
jupyter notebook main_projection_analysis.ipynb
```

3. Open the notebook and run the cells to execute the inverse projection analysis.

---

## Notes

* Python **3.12** is required to avoid compatibility issues.
* Poetry automatically manages dependencies, so **manual package installation is not needed**.
* All generated outputs (plots, analysis results) will be saved in the `results/` folder.

---
