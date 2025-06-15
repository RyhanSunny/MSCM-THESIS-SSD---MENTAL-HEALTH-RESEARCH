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