# MA 232 - Project 2: Pattern Recognition

Linear algebra project using quadratic forms and orthogonal complements to detect a 4x4 target pattern.

## Setup Python Environment

1. Create and activate a virtual environment:

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Mac / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run calculations:

```bash
python calculate.py

# if you need raw latex output for the report, run:
python calculate.py --latex
```

## Compiling the Report (`main.tex`)

Choose whichever workflow fits your setup:

### Option 1: Overleaf

1. Go to [Overleaf](https://www.overleaf.com).
2. Create a "New Project" -> "Blank Project".
3. Upload `main.tex` (or copy and paste its contents into `main.tex` on Overleaf).
4. Click **Recompile**.

### Option 2: VS Code (Local Editor)

1. Install [VS Code](https://code.visualstudio.com/).
2. Install the [LaTeX Workshop Extension](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop).
3. Ensure you have a TeX distribution installed on your machine:
   - Windows: [MiKTeX](https://miktex.org/download)
   - Mac: [MacTeX](https://www.tug.org/mactex/)
   - Linux: [TeX Live](https://www.tug.org/texlive/)
4. Open `main.tex` in VS Code and save to trigger auto-compile, or use `Ctrl+Alt+B` / `Cmd+Option+B`.

### Option 3: Terminal CLI

With MiKTeX or TeX Live installed, run from this folder:

```bash
pdflatex main.tex
pdflatex main.tex
```

Running it twice ensures all table numbers and references resolve properly.
