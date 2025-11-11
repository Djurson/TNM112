# Introduction

## Virtual enviroment

### 1. Create a virtual enviroment

Create a virutal enviroment

_MacOS:_

```bash
python3 -m venv venv
```

_Windows:_

```bash
python -m venv venv
```

### 2. Activate the virtual enviroment

Activate the virtual enviroment

_MacOS:_

```bash
source ./venv/bin/activate
```

_Windows:_

```bash
venv\Scripts\activate
```

### 3. Install packages

Make sure you have python and the required packages installed

```bash
pip install -r requirements.txt
```

### 4. Optional: Install the jupyter VS code package

Install the [Jupyter VS code package](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) from the marketplace and select the venv as the interperter when you try to run `.ipynb` files in vscode

## Starting the project

```bash
jupyter notebook
```

If you have the Jupyter VS code package. Run the `.ipynb` file(s)
