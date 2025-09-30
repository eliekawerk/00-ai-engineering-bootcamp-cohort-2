# Setting up your development environment for Python projects.
This document holds information about how you can set up your Python development environment for successful AI Project development.

## 1. Install Python.
You can download and install Python by visiting https://www.python.org/ and navigating to:

```Markdown
Downloads -> Choose your OS -> Downlaod the latest stable version -> Run the installer
```

You can check for successful installation by running:

```Bash
which python
```

It should show the location where currently used python is installed. 

You can also run:


```Bash
python --version
```
It shoould show python version that is currently being used.

## 2. Check if `pip` is configured.
pip is Python's default package manager, it should come with python installation, you can check by running:
```Bash
pip --version
```
It should print the version and location of pip that is configured. 
If `pip` is not found, you can install it by running:
```Bash
python -m pip install --upgrade pip
```
## 3. Install `uv`.
[uv](https://github.com/astral-sh/uv) is a Python package manager written in Rust. It is faster than pip and comes with additional project management capabilities. I highly recomend trying it out if you have not done so yet.

Install uv by running:

```Bash
pip install uv
```
You can check for successful installation by running:
```Bash
uv --version
```

## 4. Install Docker.
You can download docker by visiting https://www.docker.com/. 
```Markdown
Downloads Docker Desktop -> Choose your OS
```
In order to use docker on your machine, you will need to have Docker Desktop running. Check if the installation was successful by running:
```Bash
docker --version
```
## 5. IDE recommendations.
I highly recommend trying out [Cursor](https://www.cursor.com/) as your IDE for developing with Python. You can download it from the official website.

### Alternatives
- [Pycharm](https://www.jetbrains.com/pycharm)
- [VSCode](https://code.visualstudio.com/)

You will be able to run your Notebooks in all of the above IDEs directly, but if you prefer dedicated IDEs for Notebooks, you can use:
- Jupyter Lab - If you have set up uv in the previous step, start Jupyter Lab by running:
```Bash
uv run --with jupyter jupyter lab
```
- Google Colab - you can start your notebooks in Google Colab mode by uploading your notebooks to your Google Drive and running them via Google Colab Chrome application.

## 6. initiating uv project.
To bootstrap the initial files required by uv Python project managemet:
- Navigate to your project directory.
- Run:
```Bash
uv init --python==3.12
```
This will create pyproject.toml and .python-version files, we will use them to manage the project dependencies.

## 7. Create a virtual environment for the project.
To create a virtual environment to be used in the project, run:
```Bash
uv venv .venv --python 3.12
```
You can activate the environment by running:
```Bash
source .venv/bin/activate
```
Any libraries you install will now be installed locally in the active virtual environment.