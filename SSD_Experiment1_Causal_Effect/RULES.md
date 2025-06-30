# AI Governance Rules: Python Healthcare Research
*Optimized for SSD/CPCSSN on Windows*

## 🎯 Core Principles
```python
# Every script must follow
SEED = 42  # Reproducibility
DATA_DIR = Path(__file__).parent / 'data'  # No hardcoded paths
MAX_LINES_PER_FILE = 1500  # Stay under token limits
```

## 📂 File Organization Rules

### **Stop File Explosion**
```python
# ❌ AI creates new files
analysis_v1.py
analysis_v2.py
analysis_final.py
analysis_final_FINAL.py

# ✅ Use ONE file + git
analysis.py  # git handles versions
# Or if needed: analysis_20250315.py (ONE dated backup)
```

### **Module Size Limits**
```python
# Each .py file should be:
# - Under 1500 lines (≈20k tokens)
# - Single responsibility
# - Named by function, not version

# ❌ Bad structure
everything.py  # 5000 lines

# ✅ Good structure
src/
  ├── data_loader.py      # ~500 lines: loading only
  ├── preprocessor.py     # ~800 lines: cleaning only  
  ├── feature_engineer.py # ~600 lines: feature creation
  └── models.py          # ~700 lines: ML models
```

### **Documentation Control**
```python
# ❌ AI generates
README.md
README_updated.md
project_notes.md
analysis_notes_v3.md

# ✅ Enforce structure
docs/
  ├── README.md          # Project overview ONLY
  ├── CHANGELOG.md       # All changes here
  └── decisions.md       # Research decisions
  
# Everything else → code comments
```

## 🔍 AI Search & Context Rules

### **Before Creating ANY File**
```python
# 1. Use SERENA MCP to search existing code
@ai_instruction
def before_coding():
    """
    ALWAYS run first:
    - Search for existing implementation
    - Check if function already exists
    - Find similar patterns
    """
    # Example: "Find all functions that process lab data"
    
# 2. Check context7 for latest patterns
@ai_instruction  
def check_documentation():
    """Use context7 to verify latest API/library usage"""
```

### **File Split Triggers**
```python
# When to split a file:
if any([
    line_count > 1500,
    class_count > 5,
    function_count > 20,
    mixed_responsibilities  # e.g., loading + modeling
]):
    split_file()

How to split:
Original: data_processor.py (2000 lines)
Becomes:
  ├── data_validator.py    # Input validation
  ├── data_transformer.py  # Transformations
  └── data_aggregator.py   # Aggregations

## 🚨 Common AI Pitfalls → Solutions

### 1. **Path Handling**
```python
# ❌ AI generates
df = pd.read_csv('C:\\Users\\data.csv')
# ✅ Always fix to
df = pd.read_csv(Path('data') / 'data.csv')
```

### 2. **Memory Overflow**
```python
# ❌ AI loads everything
df = pd.read_csv('8GB_file.csv')
# ✅ Chunk large files
for chunk in pd.read_csv('8GB_file.csv', chunksize=50000):
    process(chunk)
```

### 3. **Vague Exceptions**
```python
# ❌ AI's lazy handling
try: complex_op()
except: pass
# ✅ Specific + actionable
try: 
    complex_op()
except pd.errors.ParserError as e:
    logger.error(f"CSV parsing failed: {e}")
    raise
```

### 4. **Missing Validation**
```python
# After EVERY data operation
assert not df.empty, "DataFrame empty after merge"
assert len(df) > 100, f"Only {len(df)} rows - check filters"
assert set(required_cols).issubset(df.columns), f"Missing: {set(required_cols) - set(df.columns)}"
```


## 📏 Code Organization Rules

### **Import Management**
```python
# At the top of EVERY file
"""
Module purpose in one line.

Functions:
- load_data: Read CPCSSN files
- validate_schema: Check required columns
"""

# Standard library
import os
from pathlib import Path

# Third party
import pandas as pd
import numpy as np

# Local (relative imports)
from .config import Config
from .utils import safe_log
```

### **Function Limits**
```python
# Each function < 50 lines
# Each class < 200 lines
# Each file < 1500 lines

# ❌ Monster function
def analyze_everything(df):
    # 500 lines of code...
    
# ✅ Decomposed
def load_ssd_data(path: Path) -> pd.DataFrame:
    """30 lines"""
    
def calculate_severity(df: pd.DataFrame) -> pd.Series:
    """40 lines"""
    
def generate_report(severity: pd.Series) -> None:
    """35 lines"""
```

### **Namespace Control**
```python
# ❌ Global namespace pollution
from utils import *
from config import *

# ✅ Explicit imports
from utils import validate_dataframe, safe_log
from config import SYMPTOM_CODES, MIN_VISITS
```

## 🔧 Quick Fixes for AI Code

| AI Generates | You Fix To |
|--------------|------------|
| Creates `analysis_v2.py` | Update `analysis.py` + git commit |
| Giant 3000-line file | Split by responsibility |
| New README variant | Update existing `docs/README.md` |
| Inline magic numbers | Move to `config.py` |
| Recreates existing function | Search first with SERENA |
| Outdated library usage | Check context7 for latest |

## 📊 Research-Specific Rules

### Config Pattern (No Magic Numbers)
```python
@dataclass
class Config:
    # DSM-5 thresholds
    MIN_ANXIETY_MONTHS: int = 6
    NORMAL_LAB_COUNT: int = 3
    # ICD-9 ranges
    SYMPTOM_CODES: range = range(780, 790)
    # Paths
    OUTPUT_DIR: Path = Path('output')
```

### Healthcare Privacy
```python
# Logging wrapper
def safe_log(msg, patient_id=None):
    """Never expose identifiers"""
    if patient_id:
        msg = msg.replace(str(patient_id), f"patient_{hash(patient_id) % 10000}")
    logger.info(msg)
```

## 🧹 Refactoring Triggers

```python
# Automatic refactor when:
triggers = {
    'file_too_large': 'lines > 1500',
    'duplicate_code': 'same logic in 3+ places',
    'mixed_concerns': 'loading + analysis in same file',
    'version_explosion': 'file_v2.py exists',
    'unclear_purpose': 'no docstring or mixed functionality'
}

Refactor pattern:
1. Extract functions to utils.py
2. Split by data flow stage
3. Delete old versions
4. Update imports
```

## ⚡ Performance + Organization

### Import Optimization
```python
# ❌ Import everything
import pandas as pd
import numpy as np  # Even if unused

# ✅ Import only what's needed
from pandas import DataFrame, Series
from numpy import where, nan
```

### Lazy Loading
```python
# For optional heavy dependencies
def use_torch_model():
    import torch  # Only import when needed
    return torch.load('model.pt')
```

## 🎨 Final Rules

1. **One file per purpose** → No v2, v3 versions
2. **Under 1500 lines** → Split if larger
3. **Search before create** → Use SERENA MCP
4. **Check latest docs** → Use context7
5. **No duplicate files** → Git handles versions
6. **Clear file names** → `data_loader.py` not `stuff.py`
7. **Document in code** → Not scattered .md files
8. **Archive don't duplicate** → Old code to `/archive`
9. **Import explicitly** → No `import *`
10. **Refactor early** → Don't wait for 3000 lines

---
# Analysis Rules and Guidelines
**Personal commitment to research integrity**

## Core Principles

### 1. Never Fabricate or Assume Data
- **ALWAYS** trace every statistic back to its source
- **NEVER** report numbers without verifying where they came from
- **ALWAYS** distinguish between:
  - Actual measured data
  - Calculated/derived values (show the calculation)
  - Proxy estimates (explain the methodology)
  - Literature-based estimates (provide citation)

### 2. Transparency in Reporting
When reporting any statistic:
1. State the exact value
2. Explain how it was calculated/obtained
3. Note any limitations or assumptions
4. Provide verifiable source or calculation method

### 3. Data Verification Checklist
Before reporting any number, ask:
- [ ] Where did this come from?
- [ ] Can I show the exact calculation or source?
- [ ] Is this an estimate, proxy, or actual measurement?
- [ ] Have I clearly labeled what type of data this is?
- [ ] Can someone else verify this number?

### 4. Examples of Good Practice

**✅ GOOD:**
"Medical costs averaged $425 per patient, calculated as proxy using Ontario billing codes: 4.74 visits × $75/visit + 0.38 referrals × $180/referral (Source: config.yaml lines 117-122)"

**❌ BAD:**
"Medical costs averaged $425 per patient"

**✅ GOOD:**
"We identified 143,579 patients (55.9%) with SSD patterns based on our exposure criteria (see 02_exposure_flag.py output)"

**❌ BAD:**
"About half of patients showed SSD patterns"

### 5. Citation Requirements
Always provide:
- For data: File path and line numbers or query used
- For calculations: Formula and input values
- For literature: Full citation with page numbers
- For estimates: Methodology and assumptions

### 6. Ongoing Verification
- Re-check numbers when updating reports
- Document any changes in calculations
- Note when preliminary findings are updated
- Keep audit trail of all analyses

## Personal Accountability
I commit to following these rules throughout the analysis to ensure research integrity and reproducibility. Every claim will be backed by verifiable evidence.