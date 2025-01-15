# End-to-End AI Workflow for Automated Multimodal Medical Image Reporting (AMMIR)

![fig](docs/figures/ammir.svg)

AMMIR is a Python library designed to benchmark end-to-end AI workflows, covering aspects such as model development, testing, training, and evaluation across both local and cloud platforms like Amazon Web Services. It also incorporates best software practices, aiming to align with medical software standards (ISO 62304).


## :nut_and_bolt: Installation
### Dev installation
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment
uv pip install -e ".[test,learning]" # Install the package in editable mode
uv pip list --verbose #check versions
pre-commit run -a #pre-commit hooks
```
See further details for installation [here](docs).

### :nut_and_bolt: Model development 
* [unit_test_simple_ml_pipeline.bash](scripts/unit_test_simple_ml_pipeline.bash) using [d](tests/test_simple_ml_pipeline.py)
```
bash scripts/unit_test_simple_ml_pipeline.bash
```

## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) (or [here](https://github.com/mxochicale/tools/blob/main/github/SSH.md))
* Clone the repository by typing (or copying) the following line in a terminal at your selected path in your machine:
```
cd && mkdir -p $HOME/repositories && cd  $HOME/repositories
git clone git@github.com:UCL/ammir.git
```

