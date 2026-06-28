# The Poetic Plural in Wartime Ukraine: “Someday we’ll become a people” 

Data and code for a within-author study of how first- and second-person pronoun use shifts in contemporary Ukrainian poetry across the 2022 full-scale invasion. This repository accompanies the article by Junyu Jiang and Amelia Glaser (U. C. San Diego) and reproduces every figure, table, and statistic reported in the paper.

## Overview

We ask whether Ukrainian poets move from the singular **"I"** toward the collective **"we"** after the full-scale invasion. Working within a curated corpus of **1,601 poems posted to Facebook by 33 poets (2014–2025)**, we compare
each poet to their own pre-war baseline rather than comparing poets to one another.

The central result is that a **flat corpus-level average masks a sharp internal split**: the corpus as a whole shows no reliable shift, but the largest moves toward the plural **"we"** are concentrated among poets who **remained physically in Ukraine** after 2022, while poets who evacuated drift in the opposite direction. At the level of grammar, the wartime corpus also sees the deontic verb **мусити ("must")** emerge as a head governing the first-person plural — a "we who must" that is essentially absent before 2022.

## Key findings

- A within-author design across 33 poets yields **no reliable period shift at the corpus level** for any pronoun cell.
- That null **masks biographical polarization**: collective "we" rises most among poets who stayed in Ukraine and falls among those who left, lining the two ends of the distribution up with poets' wartime locations.
- **мусити ("must")** enters the first-person-plural cell from zero pre-war tokens to thirteen in wartime — the clearest single grammatical signal of a shift from lyric "I" to an obligated collective "we."

## Repository structure

> Paths below reflect the intended layout — adjust to match the actual files.

```
.
├── data/
│   ├── poems/                # Corpus of 1,601 poems (text + metadata)
│   ├── biographical/         # Per-poet metadata (language, 2022 location, etc.)
│   └── annotations/          # Pronoun-cell labels and validation sample
├── src/
│   ├── 01_preprocess.py      # Cleaning, stanza segmentation, language tagging
│   ├── 02_detect_pronouns.py # Stanza dependency parse + pro-drop recovery
│   ├── 03_models.R           # Poem-level negative-binomial / Bayesian models
│   └── 04_collocations.py    # Dependency-parsed head-lemma collocation analysis
├── figures/                  # Generated figures (caterpillar, drift, collocates)
├── tables/                   # Generated tables (Table 1, etc.)
└── README.md
```

## Data

The corpus is drawn from the **Contemporary Ukrainian Poetry Archive**, an ongoing catalogue of poems posted to Facebook. 

## Method

1. **Pronoun detection (stanza level).** Each stanza is dependency-parsed with **Stanza**; because Ukrainian and Russian are null-subject languages, impliedsubjects are recovered with an LLM-assisted pro-drop step before pronoun cells (1sg, 1pl, 2sg, 2pl) are counted.
2. **Modeling (poem level).** Per-poem pronoun counts feed **poem-level negative-binomial / hierarchical Bayesian within-author models**, with poem token count as exposure and a period term for pre- vs. post-2022.
3. **Robustness.** Results are checked across exposure definitions, period codings, two model families, and a specification curve before any effect is read as real.
4. **Collocations.** A dependency-parsed, period-differential collocation analysis identifies which head verbs govern "we" before vs. during the war.
