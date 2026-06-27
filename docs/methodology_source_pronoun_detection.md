# Source-Side 1st/2nd-Person Pronoun Detection (Ukrainian / Russian)

Methodology + prototype for detecting the four morphological cells
`{1sg, 1pl, 2sg, 2pl}` **directly from the Ukrainian/Russian source text**, with no
English-translation intermediary. This replaces the translate-to-Shakespearean-English-
then-tag approach in `src/01_annotation_gpt_annotation.py`.

Companion code: `src/01_annotation_source_pronoun_detection.py`. Two engines:
**v1** (Stanza `tokenize,pos,lemma` + sentence-level heuristic, the original baseline)
and **v2** (adds `depparse`, highest precision — see Part D). Run the full comparison:

```bash
PYTHONPATH=src python src/01_annotation_source_pronoun_detection.py \
    --sample 50 --seed 42 --compare-modes v1,v2,gpt
```

**Headline result** (vs the 25-stanza hand gold, `data/gold/pronoun_gold_25_stanzas.csv`):
v2 4-cell F1 **0.933** vs v1 0.860 (**+7.3 pp**) vs GPT 0.444; v2 precision 0.992.
GPT recovers **zero** pro-drop (implied F1 = 0.000), confirming the motivation.

---

## 0. Scope (as fixed by the brief)

- **Cells:** only `1sg / 1pl / 2sg / 2pl` (personal pronouns + possessive determiners
  + recovered pro-drop subjects).
- **Languages:** rows whose `language` is exactly `Ukrainian` or `Russian`. Code-switching
  rows (`"Russian, Ukrainian"`, …), Qirimli/Crimean Tatar, Polish, English are excluded.
- **`ви`/`вы` = morphological `2pl`** always. No polite-singular vs true-plural split here
  (that lives downstream in `pronoun_encoding.poem_person_cell_column` via `vy_register`).
- **No 3rd person, impersonal, referent, inclusivity, polyphony.**
- **No LLM/translation** anywhere in the detection path. Rules + morphology only.
- The Russian column quirk: in `pronoun_annotation.csv` the source text lives in
  `stanza_ukr` **even for Russian rows** (historical column name). The pipeline is selected
  by the `language` value, never by column name.

---

## Part A — Methodology

### A.1 Three candidate architectures (and the choice)

| # | Architecture | Explicit detection | Pro-drop | Pros | Cons |
|---|---|---|---|---|---|
| 1 | **Dictionary-first** | hand-built wordform tables per case/number | verb-suffix regex (`-мо`, `-еш`, `-в/-ла/-ли`) | zero model deps; fully transparent; fast | brittle on poetic spelling; suffix rules are noisy; re-implements a POS tagger badly |
| 2 | **spaCy + Stanza hybrid** | spaCy `*_core_news_sm` `PRON`/`DET` morph | Stanza finite-verb feats + depparse `nsubj` | two taggers cross-check; depparse gives true subject attachment | **`uk_core_news_sm`/`ru_core_news_sm` are not installed here**; two backends to maintain; depparse is slow |
| 3 | **Stanza-only, rule-arbitrated** ✅ | Stanza `PRON`(`PronType=Prs`)/`DET`(`Poss=Yes`) | Stanza finite-verb `Person`+`Number`; sentence-level subject-suppression heuristic | single backend that is actually installed (uk+ru); lemma + feats are exactly what we need; dictionary kept only as a tiny lemma→cell map | depends on Stanza tagging quality on verse; no full dep-parse (mitigated, see A.4) |

**Chosen: Architecture 3 (Stanza-only).** Decisive factor: spaCy's Ukrainian/Russian models
are absent in this environment, while Stanza's `uk` and `ru` models load and already emit
`Person`, `Number`, `Case`, `PronType`, `Poss`, `Tense`, `Mood`, and lemma — every signal
the four cells need. Adding spaCy would mean maintaining and reconciling two taggers for no
new information. A small dictionary survives only as the possessive-lemma→cell map (A.3).

### A.2 Overall flow

```
stanza text ──► Stanza(tokenize,pos,lemma) ──► per sentence:
    pass 1: collect explicit nominative personal-pronoun cells   (for suppression)
    pass 2: emit detections
        ├─ explicit personal pronoun   (PRON, PronType=Prs, Person∈{1,2})
        ├─ explicit possessive det     (DET/ADJ, Poss=Yes, Person∈{1,2})
        └─ implied verb subject        (finite verb, Person∈{1,2}, not suppressed)
                                        past-tense → AMBIGUOUS_PAST (excluded)
```

- **Unit of analysis = stanza** (`poem_id` × `stanza_index`), matching the GPT table.
  Stanza's sentence splitter handles multi-sentence stanzas and verse line breaks; the
  subject-suppression heuristic (A.4) is scoped to each *sentence*, not the whole stanza, so
  a dropped subject in one line is not masked by an explicit pronoun in another.
- **Pipelines:** Stanza `uk` and `ru`, processors `tokenize,pos,lemma`, cached one per
  language. No `depparse` (see A.4 for why and how subject attachment is approximated).

### A.3 Explicit pronoun detection

Two token classes, mapped to a cell by different rules:

1. **Personal pronouns** — `upos == PRON` and `PronType == Prs` and `Person ∈ {1,2}`.
   Cell comes from the **pronoun's own** `Person` + `Number` feats
   (`Person=1,Number=Sing → 1sg`, etc.). This covers every case form, because Stanza
   lemmatises oblique forms back to the nominative lemma (`мене/мені/мною → я`,
   `тебе/тобі → ти`, `нас/нам → ми`, `вас/вам → ви`) and keeps `Person`/`Number` on the
   token. No wordform table needed.

2. **Possessive determiners** — `upos ∈ {DET, ADJ}` and `Poss == Yes` and `Person ∈ {1,2}`.
   Cell comes from the **lemma**, *not* the feats: a possessive's `Number` agrees with the
   *possessed noun* (`наш дім` is `Number=Sing`), so feats would give the wrong number.
   The possessor's number is lexical, encoded in the lemma:

   | cell | Ukrainian lemma | Russian lemma |
   |------|------------------|----------------|
   | 1sg | мій | мой |
   | 1pl | наш | наш |
   | 2sg | твій | твой |
   | 2pl | ваш | ваш |

   This matches the GPT convention of counting `my`→1st, `our`→1pl, etc.

**Why feats/lemma and not a wordform dictionary:** a morphological analyser generalises to
unseen case/number/gender combinations and poetic orthography; a static table does not. The
dictionary is reduced to the 8-row possessive map above.

**`-ся/-сь` reflexives and `-мо` (1pl):** handled automatically. Stanza tags `-мо` verbs as
`Person=1|Number=Plur` (counted as a 1pl *implied* event, not a separate pronoun). Reflexive
`-ся` verbs that are *personal* keep their `Person` feat and are counted normally; *impersonal*
`-ся` constructions (`хочеться`) get `Person=3`/no person and are correctly skipped.

### A.4 Pro-drop recovery

A finite verb emits one **implied-subject event** when it carries `Person ∈ {1,2}` and has no
co-referent explicit subject (A.4.3). "Finite verb" = `upos==VERB` excluding
`VerbForm ∈ {Inf, Conv, Part}` (reusing the logic in
`src/utils/finite_verb_exposure.is_finite_verb`).

#### A.4.1 By tense / mood

| form | feats available | rule | cell |
|------|------------------|------|------|
| Present / Future indicative | `Person` + `Number` | direct map | `1sg/1pl/2sg/2pl` |
| Imperative (`Mood=Imp`) | `Person=2` (+ unreliable `Number`) | map; flag number | `2sg/2pl` + `IMPERATIVE_NUMBER_UNCERTAIN` |
| **Past (`Tense=Past`)** | `Gender` + `Number`, **no `Person`** | **do not guess** | `""` (excluded) + `AMBIGUOUS_PAST` |

Past tense is the core difficulty: `-в/-ла/-ло/-ли` mark gender+number but **not person**
(`писав` = "I/you/he wrote"). Per the brief's instruction *"宁可 flag 为 ambiguous 并 exclude…
也不要 silent guess"*, past-tense verbs are emitted as **diagnostic rows** carrying a
candidate set (`1sg|2sg|3sg` for singular, `1pl|2pl|3pl` for plural) but with an **empty
`cell`**, so they never enter the primary counts. A later context-heuristic or human pass can
promote them; the conservative default never invents a person.

Note imperative `Number` is unreliable in practice — Stanza tagged the plural imperative
`Любіть`/`даруйте` as `Number=Sing`. We keep the tagged number but flag it so downstream can
decide whether 2sg/2pl imperative splitting is trustworthy.

#### A.4.2 Forms that must NOT trigger a count

Infinitives, converbs (`-чи`), participles/passives (`VerbForm=Part`), impersonal `-ся`,
and any 3rd-person or person-less finite form. All are excluded because they never satisfy
`Person ∈ {1,2}` after the `VerbForm` filter.

#### A.4.3 Suppressing double counts (no depparse)

Without `depparse` we cannot read the true `nsubj` edge, so we approximate: within each
**sentence** we collect the set of cells expressed by an **explicit nominative personal
pronoun** (`PRON`, `Case=Nom`, `Person∈{1,2}`). A finite verb whose implied cell is already
in that set is treated as having an explicit subject and is **not** counted again (`Я іду`
→ one explicit `1sg`, the verb `іду` is suppressed). This is conservative (it can over-
suppress in a sentence that genuinely has two same-cell clauses sharing no subject), which is
the safe direction for *not over-counting*. Upgrading to `depparse` for exact `nsubj`
attachment is the obvious production refinement; it roughly doubles Stanza runtime.

#### A.4.4 `is_pro_drop = True` definition

True iff the event originates from a finite verb with no suppressing explicit subject
(`detection_method == implied_verb` and a non-empty cell). Possessives and oblique pronoun
objects are **never** pro-drop (consistent with the existing manual convention).

### A.5 Known limitations & `qa_flag` design

| flag | meaning | counted? |
|------|---------|----------|
| `OK` | confident explicit or present/future implied | yes |
| `IMPERATIVE_NUMBER_UNCERTAIN` | imperative; sg/pl tag unreliable | yes (cell kept) |
| `AMBIGUOUS_PAST(gender=…)` | past-tense verb, person unrecoverable | **no** (cell blank) |
| `NO_VERB_NUMBER` | finite 1/2-person verb missing `Number` | **no** |

Observed tagger-level errors on verse (surfaced honestly, not hidden):
- POS/lemmatisation slips on poetic tokens — e.g. the noun `Полжизни` mis-tagged as an
  imperative verb (false `2sg`); a clipped token parsed as a finite verb.
- Person mis-tag — `Вписываем` (1pl) tagged `Person=2` → spurious `2pl`.

Expected differences vs GPT:
- **Pro-drop rate jumps from ~1–2% to ~30%+ of counted events** (the central motivation;
  GPT massively undercounts dropped subjects).
- Poetic word order, ellipsis, and line breaks are handled by morphology rather than by a
  translation model's pronoun (over/under)generation.
- GPT's `qa_flag` already logged many inconsistencies (e.g. `ви`-form tagged singular `thou`);
  those classes disappear because `ви/вы` is deterministically `2pl`.

### A.6 Evaluation plan (small-sample)

1. **Auto-diff (implemented):** stratified sample (uk/ru balanced), produce per-stanza
   `src_*` vs `gpt_*` 4-cell counts and `d_*` deltas (`diff_vs_gpt.csv`). Metrics:
   exact-4-cell-match rate per language; net per-cell delta; explicit-only vs total
   (`src_expl_*` / `gpt_expl_*`); recovered-implied share.
2. **Gold subset (recommended next):** hand-annotate ~20 stanzas (10 uk / 10 ru, biased to
   high past-tense and high pro-drop) to get precision/recall for each path *independent of
   GPT*, since GPT is not ground truth. Use the `AMBIGUOUS_PAST` rows to measure how much
   signal is being conservatively withheld.
3. Stratify reporting by language and by `detection_method` (explicit_pron / explicit_poss /
   implied_verb) so explicit and pro-drop accuracy are judged separately.

Sample run (`--sample 20`, seed 42): exact-4-cell match Ukrainian 60%, Russian 50%; net
deltas within ±2 per cell; **16/47 counted events (34%) were recovered pro-drop**; 63
past-tense events conservatively excluded as `AMBIGUOUS_PAST`.

---

## Part C — Downstream interface

The new detector produces the same conceptual unit (a per-stanza pronoun/implied event with
`person`/`number`) so downstream cell-counting is reusable.

### C.1 Field mapping to the existing schema

| existing GPT column | new source column | note |
|---|---|---|
| `poem_id, author, language, year, temporal_period, stanza_index` | identical | carried through |
| `stanza_ukr` | (input only) | source text; not re-emitted per token |
| `person` (`1st/2nd`) | `person` | same encoding (`AMBIGUOUS_PAST` rows blank) |
| `number` (`Singular/Plural`) | `number` | same encoding |
| `is_pro_drop` | `is_pro_drop` | now meaningfully populated |
| `source_mapping` | `lemma` (+ `surface_form`) | lemma replaces the dictionary form |
| `qa_flag` | `qa_flag` | new flag vocabulary (A.5) |
| `pronoun_word`, `stanza_en`, `full_shakespeare_text` | **dropped** | English artefacts, no longer needed |
| — | `cell`, `detection_method`, `governing_verb`, `verb_tense`, `candidates` | new diagnostics |

### C.2 How counting reuses existing utils

`src/utils/pronoun_encoding.py` keys off `person` (`1st/2nd/3rd`) + `number`
(`Singular/Plural`); the detector emits exactly those, so
`pronoun_class_sixway_column` and `poem_person_cell_column` work **unchanged** on the new
token table. `poem_cell_counts.py` and the Q1 pipeline (`02_modeling_q1_per_cell_glm.py`)
consume the resulting per-poem cell counts as before.

### C.3 Breaking changes for the migration

1. Only rows with non-empty `cell` enter counts; `AMBIGUOUS_PAST` rows are diagnostics —
   downstream must filter `cell != ""` (or `qa_flag == OK` ∪ imperative) before aggregating.
2. The `2pl` polite-singular split still happens downstream via `vy_register`; the detector
   emits plain morphological `2pl`. If `vy_register` was sourced from the GPT English `thou/ye`
   signal, it must be re-derived (e.g. a separate addressee-number heuristic) — the detector
   deliberately does not infer politeness.
3. Cell totals will shift up (pro-drop recovery) and slightly around (tagger errors).
   Re-run Q1/Q2 offsets; `n_finite_verbs` exposure (`00e_compute_finite_verb_exposure.py`)
   already uses the same `is_finite_verb` definition, so the offset and the numerator are now
   methodologically consistent.

---

## Part D — v2 engine (depparse, highest precision)

v2 keeps the v1 cell taxonomy and counting model but replaces the
heuristics with dependency-grounded rules (Stanza `tokenize,pos,lemma,depparse`).
v1 is preserved verbatim as `--mode v1` for baseline comparison.

### D.1 What v2 changes

1. **Exact `nsubj` attribution.** A finite verb is pro-drop *only* when it has no
   `nsubj` child on its own dependency node (for analytic futures, on the verb the
   `буду`-AUX supports). This replaces v1's sentence-level "is any matching-cell
   nominative pronoun present?" heuristic, which over-suppressed across clauses.
2. **Explicit-pronoun policy = count all roles.** Personal pronouns are counted in
   subject *and* oblique positions (object/iobj/obl), each tagged with `syntactic_role`
   (from its deprel). This matches the GPT convention (тебе/мене are real events) and
   makes the role auditable rather than implicit.
3. **Possessives by lemma, Person-feat-free.** Russian `мой/наш/твой` carry `Poss=Yes`
   but **no `Person` feat** in Stanza (Ukrainian `наш` does). v2 maps possessives by
   lemma regardless of feats — this single fix took explicit recall from 0.917 to 1.000.
4. **Analytic future.** `буду/будемо … + infinitive`: Stanza tags `буду` as
   `upos=AUX deprel=aux`; v2 treats an AUX with `Person∈{1,2}` as the implied-subject
   carrier (v1 missed every analytic future).
5. **Imperative number fix.** Stanza frequently mis-tags plural imperatives as singular.
   v2 overrides `Number` from the surface suffix (`-те/-іть/-їть` → plural), after
   stripping the reflexive particle (`дивуйтеся → дивуйте`). Flagged
   `IMPERATIVE_NUMBER_FIXED`.
6. **Past-tense promotion (no silent guess).** Past forms carry no `Person`. By default
   they stay excluded (`AMBIGUOUS_PAST`), but v2 *promotes* one to a cell when a
   coordinated (`conj`) clause supplies the person **and the number agrees**:
   `PAST_PROMOTED_CONJ_PERSON` (head is a present/future/imperative verb of the same
   number) or `PAST_PROMOTED_CONJ_NSUBJ` (head past but with an overt 1/2-person
   pronoun subject). The number-agreement guard rejects impersonal `було/было` under a
   plural `ми/мы`. Every past form (excluded or promoted) is logged in
   `ambiguous_past_audit.csv`.
7. **False-positive filter.** A finite VERB sitting in a *nominal* deprel slot
   (`nmod/obl/amod/fixed/flat/appos…`) is a noun-mis-tagged-as-verb and is rejected
   (`FILTERED_NONPREDICATE`, logged in `error_analysis.csv`). This removes
   `есть/еби/было/чорніє/ЗНАХОДИТЬ/були`-type mistags from the counts.

New token columns (v2): `nsubj_lemma, nsubj_deprel, verb_deprel, syntactic_role,
promotion_rule, qa_flag`.

### D.2 Gold evaluation (25 stanzas, uk 12 / ru 13)

Hand-annotated in `data/gold/pronoun_gold_25_stanzas.csv` (built by
`data/gold/build_gold_25_stanzas.py`), stratified toward high pro-drop / past-tense /
v1–GPT divergence. Counts are split explicit vs implied per cell; metrics are micro
count-based P/R/F1 (multiset agreement). **GPT is a comparison, not ground truth.**

| mode | P | R | **F1** | explicit F1 | implied F1 |
|------|------|------|--------|-------------|------------|
| v1   | 0.959 | 0.780 | 0.860 | 0.957 | 0.761 |
| **v2** | **0.992** | **0.880** | **0.933** | **1.000** | **0.863** |
| gpt  | 0.549 | 0.373 | 0.444 | 0.644 | **0.000** |

By language (total F1): v2 Ukrainian 0.895 (v1 0.845), v2 Russian 0.955 (v1 0.885).
Success criteria met: v2 − v1 = **+7.3 pp** (target ≥ +5 pp); v2 precision 0.992 with a
**single residual false positive** (`гав`, the onomatopoeia "woof", which Stanza tags as
an imperative verb in a coordinated predicate slot — not a noun-in-nominal-slot case, so
the deprel filter cannot catch it).

### D.3 Residual recall losses (all honest, conservative-policy)

The ~12 % implied-recall gap is dominated by, in order:
- **Imperatives mis-tagged as past** by Stanza (`Загни`, `поправ`, `постав`) → excluded as
  `AMBIGUOUS_PAST`. Their `-в/-и` surfaces are homographic with past forms, so promoting
  them would require guessing; we don't.
- **Genuine past 1/2 subjects not coordinated** (`Знайшла`=I found, `лежав`=I lay,
  `Стыдилась`=I was ashamed) — sit in `root/advcl/ccomp`, no `conj` evidence → excluded.
- **`nsubj` misparses** where an object is tagged subject (`долі` as nsubj of `малюю`)
  → the verb is wrongly suppressed.
These are visible in `ambiguous_past_audit.csv`; a future context model or 20-stanza human
pass could promote them, but the default never invents a person.

---

## Full-corpus scaling (`--full`)

- ~7,081 in-scope stanzas; v1 (`tokenize,pos,lemma`) on CPU ≈ 15–40 min; v2 (+`depparse`)
  ≈ 2× that.
- Production: (1) batch documents / use GPU; (2) checkpoint per `poem_id` to parquet shards
  and skip completed IDs on resume; (3) parallelise across poems with a process pool
  (pipelines are not thread-safe — one pipeline per worker).
