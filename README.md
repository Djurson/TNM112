# Machine Learning for Social Media labs

## 📝 Project Setup Guide

Follow these steps to clone the project, set up a Python environment, and install all required dependencies.

### 1. Clone the Repository

If you have a folder:

```bash
# Navigate to where you want the project
cd /path/to/TNM108

# Clone into the current folder (must be empty)
git clone https://github.com/Djurson/TNM112.git
```

If you don’t already have a folder prepared:

```bash
# Clone into a new folder MLSM-labs
git clone https://github.com/Djurson/TNM112.git
cd MLSM-labs
```

### 2. Create a Virtual Environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\activate
```

This creates a folder called venv and activates it so packages install locally.

### 3. Install Project Requirements

Once the virtual environment is active:

```bash
pip install -r requirements.txt
```

This installs Pandas, NumPy, SciPy, scikit-learn, seaborn, matplotlib and any other listed dependencies.

### 4. Select the Interpreter in Your Editor

**In VS Code:**

1. Open the project folder.
2. Press `⇧⌘P` (Mac) or `Ctrl+Shift+P` (Windows/Linux) to open the Command Palette.
3. Search for “Python: Select Interpreter”.
4. Browse to the `venv/bin/python` (Mac/Linux) or `venv\Scripts\python.exe` (Windows) and select it.

**In PyCharm:**

1. Go to **Settings → Project → Python Interpreter**.
2. Click the gear icon ⚙️ or `Add interpreter` button → **Add** or **Add local interpreter...**
3. Choose **Existing environment**.
4. Browse to the interpreter inside your venv and select it:

   **On mac/linux:**

   ```bash
   # .venv/bin/python
    /path/to/your/project/venv/bin/python3
    ```

   **On windows:**

   ```powershell
   # .venv\Scripts\python.exe
    C:\path\to\your\project\venv\Scripts\python.exe
    ```

   If you’re unsure of the path, run this in Terminal inside your project:

   ```bash
   which python3
   ```

### 5. Running the Code

#### Running from the terminal

Activate the virtual environment first:

```bash
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

Then run any script:

```bash
python lab1/part1/my_script.py
```

#### TL;DR

- **Terminal (outside IDE):** You must `source venv/bin/activate` every time before running your scripts.
- **Inside IDE (Run button):** You do **not** need to activate manually — just select the interpreter once.
- **Integrated terminal:** Depends on settings; often auto-activates if interpreter is selected.

### 6. Updating Dependencies

If new packages are added to the project:

Add package name to requirements.txt or

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

Then co-contributors run:

```bash
pip install -r requirements.txt
```
