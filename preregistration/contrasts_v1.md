# Confirmatory Contrasts v1

Signed date: 2026-05-05

Model family (poem-level 4-cell long format):

`y ~ person * number * C(period3, Treatment("P2_2019_21")) + C(author)`

Coding:
- `person=1` for first person (1sg/1pl), `person=0` for second person (2sg/2pl)
- `number=1` for plural (1pl/2pl), `number=0` for singular (1sg/2sg)
- Period reference is `P2_2019_21`

Confirmatory contrasts (BH correction only within these six tests):

1. `P3_vs_P2_2sg_cell_shift`
2. `P3_vs_P2_1pl_cell_shift`
3. `P3_vs_P2_person_x_number`
4. `P1_vs_P2_2sg_cell_shift`
5. `P1_vs_P2_1pl_cell_shift`
6. `P1_vs_P2_person_x_number`

Linear-weight definitions for cell-level shifts:

- `P3_vs_P2_2sg_cell_shift`:
  - `C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = +1`
  - `number:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = 0`
  - `person:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = +1`
  - `person:number:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = 0`

- `P3_vs_P2_1pl_cell_shift`:
  - `C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = +1`
  - `number:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = +1`
  - `person:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = 0`
  - `person:number:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = 0`

- `P1_vs_P2_2sg_cell_shift`:
  - `C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = +1`
  - `number:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = 0`
  - `person:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = +1`
  - `person:number:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = 0`

- `P1_vs_P2_1pl_cell_shift`:
  - `C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = +1`
  - `number:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = +1`
  - `person:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = 0`
  - `person:number:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = 0`

Interaction terms:
- `P3_vs_P2_person_x_number`:
  - `person:number:C(period3, Treatment("P2_2019_21"))[T.P3_2022plus] = +1`
- `P1_vs_P2_person_x_number`:
  - `person:number:C(period3, Treatment("P2_2019_21"))[T.P1_2014_18] = +1`

Pre-specified sensitivity analyses:
- drop-2014
- drop-switchers (`Iya Kiva`, `Andrij Bondar`, `Alex Averbuch`, `Olena Boryshpolets`)
- leave-one-author-out over included roster
- strict roster threshold (`min_per_period >= 8`)
