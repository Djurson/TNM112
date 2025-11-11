# Introduction

## Enviroments

<details>
<summary>Virtual enviroment (VS code)</summary>

### 1. Install the jupyter VS code package

Make sure you have the [Jupyter VS code package](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) installed

### 2. Create a virtual enviroment

Create a virutal enviroment

_MacOS:_

```bash
python3 -m venv venv
```

_Windows:_

```bash
python -m venv venv
```

### 3. Activate the virtual enviroment

Activate the virtual enviroment

_MacOS:_

```bash
source ./venv/bin/activate
```

_Windows:_

```bash
venv\Scripts\activate
```

### 4. Install packages

Make sure you have python and the required packages installed

```bash
pip install -r requirements.txt
```

### 5. Select the enviroment

Select the venv interperter when you try to run `.ipynb` files
</details>

<details>
    <summary>Global enviroment</summary>

Make sure you have python and the required packages installed

```bash
# MacOS
pip3 install -r requirements.txt

# Windows
pip install -r requirements.txt
```

</details>

## Starting the project

```bash
jupyter notebook
```
