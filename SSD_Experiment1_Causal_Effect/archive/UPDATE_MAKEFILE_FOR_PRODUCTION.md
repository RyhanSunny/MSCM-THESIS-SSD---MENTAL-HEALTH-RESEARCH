# Makefile Update Instructions

## Issue Found
The main `all` target includes week1-validation through week4-all but is missing week5 components.

## Required Update
In the Makefile, line 17, update the `all` target to include week5-validation:

**Current:**
```makefile
all: cohort exposure mediator outcomes confounders lab referral missing misclassification master sequential ps causal mediation temporal evalue competing-risk death-rates robustness week1-validation week2-all week3-all week4-all
```

**Should be:**
```makefile
all: cohort exposure mediator outcomes confounders lab referral missing misclassification master sequential ps causal mediation temporal evalue competing-risk death-rates robustness week1-validation week2-all week3-all week4-all week5-validation
```

This ensures that `make all` runs the complete Week 1-5 pipeline including all compliance checks.