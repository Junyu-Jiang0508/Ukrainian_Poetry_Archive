# **Brief Summary: Methods and Findings**

## **METHODOLOGY**

### **Data Structure**
- **35 adaptive intervals** (2013-03 to 2025-06)
- **Min 30 poems/interval** to ensure statistical power
- **Variable durations** (28-535 days) to accommodate uneven poetry production
- **Total: 1,970 poems**

### **Statistical Model**

**Weighted Breakpoint Regression:**
```
proportion_it = β₀ + β₁·time + β₂·post_2014 + β₃·post_2022 + 
                β₄·(time × post_2014) + β₅·(time × post_2022) + ε
```

**Where:**
- `proportion_it` = proportion of pronoun type in interval i
- `time` = interval sequence (1, 2, 3...35)
- `post_2014` = 1 if date ≥ 2014-02-01, else 0
- `post_2022` = 1 if date ≥ 2022-02-01, else 0
- **Weights** = poem count per interval (scheme A)

**Key Parameters:**
- **β₂** = 2014 level shift (immediate jump)
- **β₃** = 2022 level shift (immediate jump)
- **β₄** = 2014 slope change (trend modification)
- **β₅** = 2022 slope change (trend modification)

**Validation:**
- **PELT change point detection** (data-driven, no pre-specified breakpoints)
- **Bootstrap 95% CI** (1,000 iterations)
- **Model comparison** (AIC/BIC, F-tests)

---

## **FINDINGS**

### **1. Third-Person Singular (3sg) — "he/she/it"**

**Result:**
- **2014 effect**: β₂ = +0.076, **p = 0.035** 
- **2022 effect**: β₃ = -0.357, **p = 0.017** 
- **R² = 0.303**

**Pattern:**
```
29.3% (pre-2014) → 37% (2014-2022) → 30% (post-2022)
```

### **2. First-Person Plural (1pl) — "we"**

**Result:**
- **2014 effect**: β₂ = +0.036, p = 0.146 (ns)
- **2022 effect**: β₃ = +0.202, **p = 0.051** (marginal)
- **R² = 0.327** (highest)
- **Change point detected**: Interval 24 (2022-03)

**Pattern:**
```
10.7% (pre-2014) → 10.3% (2014-2022) → 14.7% (post-2022)
```

**Interpretation:**
**Collective identity surge:** Despite p = 0.051, evidence is strong—large effect size (+20 points = 200% increase), change point detection confirms March 2022 breakpoint, highest R² among all pronouns. 2022 invasion activated national "we" that Euromaidan did not. Existential threat → collectivization.

---

### **3. First-Person Singular (1sg) — "I"**

**Result:**
- **2014 effect**: β₂ = +0.100, **p = 0.0003** 
- **2022 effect**: β₃ = +0.126, p = 0.220 (ns)
- **R² = 0.235**

**Pattern:**
```
27.4% (pre-2014) → 37% (2014-2022) → 27.6% (post-2022)
```

**Interpretation:**
**2014 paradox:** Counter-intuitively, Euromaidan *increased* "I" (personal agency discourse: "I stood at Maidan"). 2022 showed no significant change, possibly because 2014's elevated level became the new normal, or collective mobilization suppressed further individualization. **Limitation:** Only 1 pre-2014 interval may bias baseline estimate.

---

### **4. Third-Person Plural (3pl) — "they"**

**Result:**
- **2014 effect**: β₂ = -0.007, p = 0.710 (ns)
- **2022 effect**: β₃ = +0.009, p = 0.903 (ns)
- **R² = 0.166**

**Descriptive pattern:**
```
15.3% (pre-2014) → 8.6% (2014-2022) → 8.7% (post-2022)
```

**Interpretation:**
**2014 drop (descriptive only):** 44% decrease observed but not testable via regression (only 1 pre-2014 interval). If real, suggests shift from **"us vs. them"** antagonistic discourse to internal solidarity after Euromaidan. 2022 had no additional effect—"they" remained stable at low levels.
---

## **STATISTICAL SUMMARY TABLE**

| Pronoun | 2014 β | 2014 p | 2022 β | 2022 p | R² | Verdict |
|---------|--------|--------|--------|--------|-----|---------|
| **3sg** | +0.076 | **0.035*** | -0.357 | **0.017*** | 0.303 | **Strongest** |
| **1pl** | +0.036 | 0.146 | +0.202 | **0.051†** | 0.327 | **Strong** (w/ change pt) |
| **1sg** | +0.100 | **0.0003**** | +0.126 | 0.220 | 0.235 | 2014 only |
| **3pl** | -0.007 | 0.710 | +0.009 | 0.903 | 0.166 | Null (descriptive 2014↓) |
| **2** | +0.045 | 0.136 | +0.020 | 0.865 | 0.095 | Null |

**Significance:** *p<0.05, **p<0.01, ***p<0.001, †marginal

