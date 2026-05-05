# Roster Freeze Decisions (v1)

- Decision date: 2026-05-05
- Baseline source: `outputs/13_descriptive_statistics/C_poem_perspective_derived_per_poem.csv` (post layer0 exclusions).
- Time bins fixed: P1=2014-2018, P2=2019-2021, P3=2022+.
- Author inclusion threshold fixed before modeling: >=5 poems in each period.
- Robustness roster also recorded at >=8 poems/period (n=11).
- Qirimli family exclusion applied at code-level: `Qirimli`, `Russian, Qirimli`, `Ukrainian, Qirimli`.
- Main-analysis switcher policy: keep bilingual switchers; language retained as covariate.

## Excluded Authors

### Alie Kenzhalieva

- Rationale: Category error: Crimean Tatar/Russian bilingual voice (exclude whole author).
- Decision date: 2026-05-05

## Andrij Bondar (P3 Russian anomaly) Diagnostic

- Protocol: spot-check at least 5 Russian poems in 2022+ against original post date/text metadata.
- Population in this dataset: present; spot-check below uses layer0 metadata.

| poem_id | date_posted | facebook_url | text_preview |
|---|---|---|---|
| UP1006_1 | 2022-03-23 | https://www.facebook.com/andrij.bondar/posts/pfbid0iLB3Rs5RYq76b7v9hHNX75gn2r4ES621ZgZfFDT1nZpKKL2m2oZaK6TWUtadf8igl | Сожги на площади русских классиков, Не будь ВКонтакте, убей Одноклассников, А на свиданье с любимой возьми Бензопилу для Московской Резни. 2014, 2022 |
| UP1015_1 | 2022-03-24 | https://www.facebook.com/andrij.bondar/posts/pfbid02ZpKpMQuVsXvhNDpeCMsVpheFk1akkrKrKTNht4YokY2RUFSJ92bpM3xdNZ7mk3uAl | Подросток-сын к родителю пришёл. Отец угрюм и смотрит виновато: "Ведь если б я в тебя, сынуля, не вошёл, В меня вошли бы члены членов НАТО". |
| UP691_1 | 2022-02-10 | https://www.facebook.com/andrij.bondar/posts/10151821112706599 | Стоят титушки, стоят сиротки, Рубасы в руках теребят, Потому что на сижки и водку Не хватает у этих ребят. |
| UP808_1 | 2022-03-03 | https://www.facebook.com/andrij.bondar/posts/pfbid02hBtFmoFZoA79E93r9qBJFdwzQnJkvd9RerFnTQqECMP5ScerCRHbWsixQxBHHPikl | Вот ватника взять: он ядрен и небрит, И тело его велико. Под ватником сердце тревожно стучит И нервно играет очко. Три четверти ватника ь это вода, А четверть ь аминокислоты. Немно |
| UP809_1 | 2022-03-03 | https://www.facebook.com/andrij.bondar/posts/pfbid0sXZMf73Jk6hs4XJe9xBEhFd9RKfwC3hxwgZN3N11xdai5jLAcpc1rtxNk8jsCvzhl | *** Ой ты гой еси, царь Владимир Владимирович, Гой еси и изгой, царь Владимир Владимирович. |

- Freeze decision for Bondar: **included** (no layer0 metadata evidence of translation/republication misattribution in sampled items).
- Caveat: this is a metadata-based check; full external FB verification can be appended if required by review.

## Threshold Rationale

- `>=5` preserves cross-period continuity while avoiding over-pruning persistent poets.
- `>=8` roster exported as robustness check to guard against threshold-driven conclusions.


## 2026-05-05 Post-freeze audit note (v2 modeling path)

- Discrepancy source identified: v2 modeling used analysis-feasible poem subset requiring 1/2-person pronoun availability and `n12 >= 5` per poem, which was stricter than v1 roster freeze basis.
- This filter reduces feasible cohort from 12 included authors to 9 under the same period threshold (>=5 per period).
- New feasible roster file: `outputs/15_roster_freeze/roster_v2_n12ge5_frozen.csv`.
- Authors dropped by feasibility rule: `Andrij Bondar`, `Ivan Andrusiak`, `Ludmila Khersonskaya` (plus already-excluded `Alie Kenzhalieva`).
- Refit requirement: all v2 confirmatory/typology outputs should be interpreted only after refit on the 9-author feasible roster.
