# Annotation Manual (Advanced): Pronoun Referents, Inclusivity, and Polyphony

---

## 0. General principles

* **Language**: **only Ukrainian poems**.

* **Unit**:
  * Token-level tasks: each row = one **pronoun token**.
  * Poem-level task: each poem ID = one **document**.

* **Context to read**:
  * Start from the `context` field (usually the line or sentence around the pronoun).
  * If unclear, open the full poem (`text`) for additional context.

* **Golden rule**:
  * When in doubt, **use "UNCERTAIN/AMBIGUOUS" rather than guessing**. 

### 0.1 Core annotation columns

**Token-level (pronoun table)**:

* `referent_category` – main referent (see Section 2)
* `referent_confidence` – confidence level (1–3)
* `we_inclusivity` – inclusivity type (only for 1pl "ми")
* `we_incl_confidence` – confidence level (1–3)
* `token_notes` – free text notes
* `syntactic_position` – grammatical position
* `semantic_role` – semantic role
* `discourse_function` – discourse function
* `context_window` – standardized context
* `annotation_difficulty` – task difficulty rating
* `polyphony_type` – voice structure
* `polyphony_notes` – free text notes

---

## 1. Token selection

1. **Stratified sampling strategy**:
   * Sample 400–600 tokens across time periods (2014–2021 vs 2022–2025)
   * Balance representation across themes
   * Include edge cases and typical examples

---

## 2. Annotating `referent_category`

### 2.1 Definition

> **Referent category** = the **main collective or person** that the pronoun refers to in this context.

### 2.2 Enhanced label set with temporal sensitivity

Based on Kulyk (2024) on identity shifts between 2014 and 2022:

#### **Core categories**:

1. `SELF` – individual poet/self
2. `INTIMATE` – family, close friends, intimate circle
3. `LOCAL` – local community (city, region, battalion, unit)
4. `NATION` – Ukrainian national collective
5. `STATE` – state institutions
6. `ENEMY` – Russia, occupiers, collaborators
7. `SUPRANATIONAL` – Europe, West, humanity
8. `OTHER` – other identifiable groups
9. `UNCERTAIN` – unresolvable ambiguity

**NATION subtypes**:
* `NATION_PRE2022` – defensive nationalism (2014–2021)
* `NATION_POST2022` – mobilized nationalism (2022+)

**ENEMY subtypes**:
* `ENEMY_IMPLICIT` – euphemistic references ("вони", "ті, хто прийшов")
* `ENEMY_EXPLICIT` – direct naming ("росіяни", "окупанти", "орки")

---

### 2.3 Decision procedure with algorithmic logic

```
STEP 1: Check explicit markers
  ├─ Has explicit national markers? ("українці", "народ", "нація") → NATION
  └─ No → Go to Step 2

STEP 2: Check spatial markers
  ├─ Has local spatial markers? (city names, "тут", "в нашому місті") → LOCAL
  └─ No → Go to Step 3

STEP 3: Check kinship terms
  ├─ Has kinship/family terms? ("мама", "діти", "родина") → INTIMATE
  └─ No → Go to Step 4

STEP 4: Check military context
  ├─ Unit-level? ("наша рота", "взвод") → LOCAL
  ├─ National army? ("наші захисники", "українська армія") → NATION
  └─ No clear military context → Go to Step 5

STEP 5: Check enemy markers
  ├─ Has enemy markers? ("окупанти", "вони прийшли/бомблять") → ENEMY
  └─ No → Go to Step 6

STEP 6: Assess context breadth
  ├─ Poem theme: individual/personal → Consider SELF or INTIMATE
  ├─ Poem theme: community/city → Consider LOCAL
  ├─ Poem theme: national struggle → Consider NATION
  ├─ Poem theme: global/universal → Consider SUPRANATIONAL
  └─ Cannot determine → UNCERTAIN
```

---

### 2.4 Boundary case protocols

**Boundary Case 1: LOCAL vs NATION**

* **Rule**: If specific place name + national framing ("ми всі українці") → `NATION` (national frame prioritized)
* **Rule**: If only "наше місто" + no national markers → `LOCAL`

**Example**:
* "Ми, кияни, захищаємо Україну" → `NATION` (national struggle frame)
* "Ми, кияни, відбудовуємо наше метро" → `LOCAL` (city-specific focus)

**Boundary Case 2: INTIMATE vs NATION**

* **Rule**: "наші діти" depends on overall poem theme:
  * Personal narrative about speaker's children → `INTIMATE`
  * Collective future ("діти України", "next generation") → `NATION`

**Example**:
* "Ми з дітьми ховалися в метро" (personal experience) → `INTIMATE`
* "Наші діти побачать вільну Україну" (generational narrative) → `NATION`

**Boundary Case 3: INCLUSIVE vs AMBIGUOUS (for we_inclusivity)**

* **Default rule**: No explicit second-person markers → `AMBIGUOUS`
* **Exception**: Imperative structures ("Давайте!", "Згадаймо!", "Разом ми...") → `INCLUSIVE`

---

### 2.5 Category definitions & lexical cues

#### 2.5.1 SELF

**Definition**: the pronoun refers to **the poet as an individual person**.

**Cues**:
* Personal experiences, feelings, memories ("моє дитинство", "моє тіло", "мій страх")
* "ми" used as modest/editorial "I" in highly self-centered poem
* Focus on individual consciousness, not collective
---

#### 2.5.2 INTIMATE

**Definition**: **family or very close circle** (partner, children, parents, close friends).

**Lexical cues**:
* Kinship terms: "мама, батько, діти, коханий/кохана, сестра, брат"
* Phrases: "ми з тобою", "ми з дітьми", "ми з мамою"

**Examples**:
* "Ми з тобою пережили цю ніч" → `INTIMATE`
* "Ми, наші діти і батьки, всі чекаємо на мир" → `INTIMATE`

---

#### 2.5.3 LOCAL

**Definition**: **local community or unit**, not whole nation.

**Types**:
* Geographic: city/town/village ("кияни", "львів'яни", "мешканці двору")
* Military: battalion/unit ("наша рота", "наш батальйон")
* Refugee community in specific place

**Cues**:
* Place names: "у Харкові", "у Бучі", "у Маріуполі" + "ми"
* Unit language: "наш взвод", "наші хлопці з третьої роти"

**Rule**: When LOCAL represents "this city as part of Ukraine", still code `LOCAL` (more specific than NATION).

---

#### 2.5.4 NATION

**Definition**: **Ukrainians as national collective** or Ukraine as a people (not institution).

**Lexical cues**:
* National markers: "українці", "народ", "наш народ", "наші воїни", "ми – українці", "ми – нація"
* Theme: national solidarity across regions, war as national struggle

**Examples**:
* "Ми вистоїмо, бо ми українці" → `NATION`
* "Ми переможемо, наша земля не скориться" (implicit but clear) → `NATION`

**Temporal variation** (optional annotation):
* Pre-2022: often defensive, reactive ("ми захищаємося")
* Post-2022: more mobilized, active ("ми переможемо", "ми непереможні")

---

#### 2.5.5 STATE

**Definition**: **state institutions and formal authorities** (government, president, parliament).

**Cues**:
* Institution terms: "уряд", "президент", "влада", "держава", "міністерство", "Верховна Рада"
* Critiques of "they" as "those in power"

**Examples**:
* "Ми ухвалили такий закон" (parliament context) → `STATE`
* "Вони знову нічого не зробили у Києві" (if refers to authorities) → `STATE`

---

#### 2.5.6 ENEMY

**Definition**: **Russia, Russian army, occupiers, collaborators**.

**Lexical cues**:
* Explicit: "росіяни", "окупанти", "орки", "москалі", "рашисти"
* Implicit: "вони" with strongly negative verbs in war context ("бомблять", "вбивають", "прийшли з півночі")

**Intensity subtypes** (optional):
* `ENEMY_IMPLICIT`: "вони прийшли", "ті, хто це зробив"
* `ENEMY_EXPLICIT`: "орки спалили наш дім", "російські окупанти"

**Examples**:
* "Вони прийшли вночі, вони спалили наш дім" → `ENEMY` (or `ENEMY_EXPLICIT`)
* "Ми женемо їх від нашої землі" ("їх" = Russians) → `ENEMY`

---

#### 2.5.7 SUPRANATIONAL

**Definition**: **humanity/world/Europe/West**, beyond Ukrainian national frame.

**Cues**:
* Global terms: "Європа", "світ", "людство", "люди", "ми всі на цій планеті"
* Themes: global solidarity, universal suffering, humanity

**Examples**:
* "Ми всі відповідальні за цю землю" (global environmental message) → `SUPRANATIONAL`
* "Ми, європейці, не можемо мовчати" (European collective) → `SUPRANATIONAL`

**Disambiguation**:
* If poem is about "Ukraine as European" but focuses on Ukraine → `NATION`
* If really about "all Europeans/humanity" → `SUPRANATIONAL`

---

#### 2.5.8 OTHER

**Definition**: **clearly identifiable group not covered above**.

**Examples**:
* "We poets", "we artists", "we women" (when not tied to national/local identity)
* Generic social categories ("діти війни", "біженці" as abstract category)

---

#### 2.5.9 UNCERTAIN

Use `UNCERTAIN` when:
* Cannot decide even after reading full poem
* Two+ categories equally strong, none dominates
* Poem deliberately plays on ambiguity

**Action**: Add brief explanation in `token_notes` (e.g., "could be NATION or SUPRANATIONAL; author keeps this open")

---

## 3. Annotating `we_inclusivity` (only for 1pl "ми")

### 3.1 Enhanced inclusivity scale

Based on Santulli (2020) on grammatical metaphor, use **5-level scale** for nuanced analysis:

1. `SPEAKER_EXCLUSIVE` – speaker only (editorial/modest we)
2. `SELECTIVE_EXCLUSIVE` – specific in-group excluding addressee (e.g., soldiers to civilians)
3. `AMBIVALENT` – strategically ambiguous
4. `SELECTIVE_INCLUSIVE` – conditional inclusion (e.g., "supporters of Ukraine")
5. `UNIVERSAL_INCLUSIVE` – unconditional inclusion of all humanity

### 3.2 Standard 3-level scale (for basic annotation)

For **first round of annotation**, use simplified scale:

1. `INCLUSIVE` – speaker + addressee (+ possibly others)
2. `EXCLUSIVE` – speaker + some others, **not** addressee
3. `AMBIGUOUS` – cannot be resolved / intentionally open

> **Note**: Use 5-level scale only when analyzing political rhetoric or strategic communication. Default to 3-level for poetry annotation.

---

### 3.3 Decision procedure

When annotating 1pl "ми":

1. **Identify addressee** in poem:
   * Is there "ти/ви" in second person?
   * Are there vocatives ("друзі", "брати і сестри")?
   * Is it public proclamation to "everyone"?

2. **Ask**: "Is this 'we' supposed to include that addressee?"

3. **Apply rules**:

#### INCLUSIVE

Code when: poem addresses audience and **invites them into "we"**.

**Patterns**:
* "Друзі, ми маємо триматися разом" (includes "друзі") → `INCLUSIVE`
* "Ми всі маємо пам'ятати про Бучу" → `INCLUSIVE`

**Cues**:
* "ми всі", "разом з вами", "давайте ми"
* Calls to action where reader is part of "we"

#### EXCLUSIVE

Code when: "ми" refers to **subgroup** that addressee is **not** part of.

**Patterns**:
* "Ми стоїмо на фронті, а ви спіть спокійно" (soldiers vs civilians) → `EXCLUSIVE`
* "Ми, поети, бачимо це інакше, ніж ви критики" → `EXCLUSIVE`

**Cues**:
* Contrastive structure: "ми …, а ви …"
* Spatial split: "ми тут, а ви там"

---

## 4. Enhanced linguistic features

### 4.1 Syntactic position

**Definition**: Grammatical position of pronoun in sentence.

**Labels**:
* `SUBJECT` – main clause subject ("Ми переможемо")
* `OBJECT` – direct/indirect object ("Вони бачать нас")
* `POSSESSIVE` – possessive modifier ("наша земля")
* `PREPOSITIONAL` – in prepositional phrase ("про нас", "з нами")

**Purpose**: Syntactic position correlates with discourse prominence and referent type.

---

### 4.2 Semantic role

**Definition**: Semantic role in event structure.

**Labels**:
* `AGENT` – performer of action ("Ми будуємо")
* `PATIENT` – undergoer of action ("Нас бомблять")
* `EXPERIENCER` – psychological state ("Ми відчуваємо")
* `BENEFICIARY` – recipient ("Ми отримали допомогу")

**Purpose**: Semantic roles reveal agency patterns in war discourse.

---

### 4.3 Discourse function

**Definition**: Function in discourse organization.

**Labels**:
* `TOPIC_CONTINUATION` – maintains current topic
* `TOPIC_SHIFT` – introduces new topic
* `CONTRASTIVE` – marks contrast ("А ми...")
* `EMPHATIC` – marks emphasis ("Саме ми")

**Purpose**: Discourse functions show information structure and rhetorical strategies.

---

## 5. Annotating poem-level `polyphony_type`

For each poem (unique `ID`), assign **one** label:

1. `MONOLOGIC` – single stable voice
2. `MEDIATED_POLYPHONY` – central narrator collecting/quoting other voices
3. `CHORAL_POLYPHONY` – multiple voices with equal status, minimal mediation

### 5.1 Definitions

---

### 5.2 Decision procedure

---

## Appendix: Quick reference

### A.1 Referent categories (9 core)

1. SELF – poet/individual
2. INTIMATE – family/close friends
3. LOCAL – city/community/unit
4. NATION – Ukrainian collective
5. STATE – government/institutions
6. ENEMY – Russia/occupiers
7. SUPRANATIONAL – Europe/world/humanity
8. OTHER – other identifiable groups
9. UNCERTAIN – unresolvable

### A.2 Inclusivity scale (3-level)

1. INCLUSIVE – includes addressee
2. EXCLUSIVE – excludes addressee
3. AMBIGUOUS – unclear/strategic ambiguity

### A.3 Polyphony types (3)

1. MONOLOGIC – single voice
2. MEDIATED_POLYPHONY – narrator collects voices
3. CHORAL_POLYPHONY – multiple equal voices

---