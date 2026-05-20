#!/usr/bin/env python3
"""Apply external research batch to author_covariates.csv + provenance audit."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COV = ROOT / "data" / "author_covariates.csv"
AUDIT = ROOT / "data" / "author_covariates_provenance_audit.csv"
PROV = "SOURCE_RESEARCH_WEB"
TABLE = "web_research_2026-05"

CORPUS_FIELDS = frozenset(
    {"language_corpus_p1", "language_corpus_p2", "bilingual_switcher_corpus"}
)
RESEARCH_FIELDS = frozenset(
    {
        "gender",
        "birth_year",
        "generation_cohort",
        "region_of_birth",
        "region_jan2022",
        "region_at_archive_freeze",
        "language_xlsx_primary_at_freeze",
        "notes",
        "references",
    }
)


def norm(v: str) -> str:
    v = (v or "").strip()
    return "" if v.lower() == "unknown" else v


def parse_batch(text: str) -> list[tuple[dict[str, str], str]]:
    """Parse author lines + # sources blocks."""
    entries: list[tuple[dict[str, str], str]] = []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    i = 0
    cols = [
        "author",
        "gender",
        "birth_year",
        "generation_cohort",
        "region_of_birth",
        "region_jan2022",
        "region_at_archive_freeze",
        "language_xlsx_primary_at_freeze",
        "language_corpus_p1",
        "language_corpus_p2",
        "bilingual_switcher_corpus",
        "notes",
    ]
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            i += 1
            continue
        parts = next(csv.reader([ln]))
        if len(parts) < 12:
            raise ValueError(f"Bad row ({len(parts)} fields): {ln[:80]}")
        row = {c: norm(parts[j]) for j, c in enumerate(cols[:-1])}
        if len(parts) > 12:
            row["notes"] = norm(",".join(p for p in parts[11:] if p))
        else:
            row["notes"] = norm(parts[11])
        refs = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("# sources:"):
            src = lines[i + 1]
            if "(none" not in src.lower():
                urls = re.findall(r"https?://[^\s\)|\]]+", src)
                refs = " | ".join(urls)
            i += 2
        else:
            i += 1
        row["references"] = refs
        entries.append((row, ln))
    return entries


BATCH = r"""
Boris Khersonsky,M,1950,pre_1970,west_ukraine,south_ukraine,south_ukraine,bilingual,,,,born: Chernivtsi (west_ukraine); long-time Odesa; Russian primary historically; Ukrainian publishing post-2014 → bilingual; PEN: PEN member; other locations: Rome
# sources: https://en.wikipedia.org/wiki/Borys_Khersonskyi | https://pen.org.ua/en/members/hersonskyj-borys | https://lannan.georgetown.edu/boris-kershonsky/ | https://www.thecafereview.com/summer-2024-poet-boris-khersonsky/
Ihor Mitrov,M,1991,1990s,south_ukraine,kyiv,kyiv,Ukrainian,,,,born: Kerch (south_ukraine; Kerch/crimea taxonomy conflict noted); Jan 2022: Kyiv; current: Kyiv; 95th Air Assault Brigade since Mar 2022
# sources: https://www.versopolis.com/poet/610/ihor-mitrov | https://chytomo.com/en/ihor-mitrov-a-bohemian-poet-at-war-turns-into-an-ordinary-soldier/ | https://pen.org.ua/en/ihor-mitrov-bohemnyj-poet-na-vijni-peretvoryuyetsya-u-zvychajnoho-soldata
Iryna Shuvalova,F,1986,1980s,kyiv,diaspora,diaspora,Ukrainian,,,,born: Kyiv; Jan 2022: diaspora (Nanjing/China); current: diaspora (Oslo since 2023); PhD Cambridge 2020; PEN: PEN member; other locations: Cambridge· UK
# sources: https://en.wikipedia.org/wiki/Iryna_Shuvalova | https://pen.org.ua/en/autors/shuvalova-iryna | https://www.versopolis.com/poet/556/iryna-shuvalova | https://poets.org/poet/iryna-shuvalova
Oleksandr Irvanets,M,1961,pre_1970,west_ukraine,kyiv,kyiv,Ukrainian,,,,born: Lviv; raised Rivne; Jan 2022: Kyiv (Irpin); current: Kyiv (Irpin); Bu-Ba-Bu co-founder; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Oleksandr_Irvanets | https://pen.org.ua/en/members/irvanets-oleksandr | https://www.wikidata.org/wiki/Q3489996
Serhiy Zhadan,M,1974,1970s,east_ukraine,east_ukraine,east_ukraine,Ukrainian,,,,born: Starobilsk (Luhansk); Jan 2022: east_ukraine (Kharkiv); current: east_ukraine (Kharkiv); 13th Khartia Brigade National Guard May 2024; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Serhiy_Zhadan | https://pen.org.ua/en/members/zhadan-sergij | https://imperiia.scalar.fas.harvard.edu/imperiia/serhiy-zhadan-i-----farwell-to-a-slavic-woman | https://www.wordsforwar.com/serhiy-zhadan-bio
Anton Polunin,M,1985,1980s,unknown,unknown,unknown,Ukrainian,,,,DOB 20 Jul 1985 (Smoloskyp); Kyiv Poetry Week co-founder 2016; ZSU service; birth place and residence unknown (no source)
# sources: https://www.smoloskyp.com.ua/product/volossya/ | https://nashformat.ua/products/volossya-940504 | https://arkushi.com/poety/anton-polunin/ | https://litcentr.in.ua/publ/279-1-0-17425
Julia Musakovska,F,1982,1980s,west_ukraine,west_ukraine,west_ukraine,Ukrainian,,,,born: Lviv (9 Jul 1982); Jan 2022: west_ukraine (Lviv); current: west_ukraine (Lviv); Lviv Nat'l Univ alum; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Yuliya_Musakovska | https://pen.org.ua/en/autors/musakovska-yuliya | https://www.arrowsmithpress.com/yuliya-musakovska/ | https://www.catranslation.org/person/yuliya-musakovska/
Kateryna Kalytko,F,1982,1980s,central_ukraine,central_ukraine,central_ukraine,Ukrainian,,,,born: Vinnytsia (8 Mar 1982); Jan 2022: central_ukraine; current: central_ukraine (Vinnytsia/Sarajevo); Shevchenko Prize 2023; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Kateryna_Kalytko | https://pen.org.ua/en/autors/kalytko-kateryna | https://www.wordsforwar.com/kateryna-kalytko-bio | https://www.versopolis.com/author/551/kateryna-kalytko
Iya Kiva,F,1984,1980s,east_ukraine,kyiv,west_ukraine,bilingual,,,,born: Donetsk; Jan 2022: Kyiv; current: west_ukraine (Lviv post-Feb 2022); first collection 2018 bilingual UA/RU; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Iya_Kiva | https://www.versopolis.com/poet/467/iya-kiva | https://kulturvermittlung.org/stipendiatinnen/iya-kiva/?lang=en | https://www.iwm.at/documenting-ukraine/blog/only-love-is-worth-fighting-for-iya-kiva-on-her-home-in-donetsk
Yury Solomko,unknown,,unknown,unknown,unknown,unknown,unknown,Russian,,,,PENDING: no biographical source for poet Yury Solomko; possible namesake with visual artist Yuriy Solomko (Crimea-born; Kyiv-based); xlsx Kharkiv birth unverified
# sources: https://www.artmajeur.com/yuriy-solomko | https://archive.kyivpost.com/article/guide/museums/boundaries-by-yury-solomko-58206.html
Alex Averbuch,M,1985,1980s,east_ukraine,diaspora,diaspora,bilingual,,,,born: Novoaidar (Luhansk Oblast); Israel 2001-2015; Toronto PhD 2022; asst prof U Michigan; bilingual UA/RU; latest collection predominantly Ukrainian
# sources: https://lsa.umich.edu/slavic/people/faculty/alex-averbuch.html | https://lyrik-in-transition.uni-trier.de/even/lecture-oleksandr-averbuch/ | https://ukrainianjewishencounter.org/en/relieving-the-terrible-knots-of-history-an-interview-with-alex-averbuch/ | https://homintern.soy/issues/11-9-20/frozensilent.html
Halyna Kruk,F,1974,1970s,west_ukraine,west_ukraine,west_ukraine,Ukrainian,,,,born: Lviv (30 Nov 1974); Jan 2022: west_ukraine (Lviv); current: west_ukraine (Lviv); Ivan Franko Univ professor; former VP PEN Ukraine; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Halyna_Kruk | https://www.versopolis.com/poet/207/halyna-kruk | https://www.lyrikline.org/en/authors/halyna-kruk | https://philology.lnu.edu.ua/en/employee/halyna-kruk
Andrij Bondar,M,1974,1970s,central_ukraine,kyiv,kyiv,Ukrainian,,,,born: Kamianets-Podilskyi (14 Aug 1974); Jan 2022: kyiv; current: kyiv (Vorzel suburb); PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Andriy_Bondar | https://literaturfestival.com/en/authors/andrij-bondar/ | https://www.wikidata.org/wiki/Q651264 | https://pen.org.ua/members/bondar-andrij
Mykhailo Zharzhailo,M,1988,1980s,west_ukraine,unknown,unknown,Ukrainian,,,,born: Rivne (7 Apr 1988); moved to Kyiv for studies; Wikidata also lists Lviv; PEN: PEN member; Jan 2022/current residence unknown (hard-stop)
# sources: https://en.wikipedia.org/wiki/Mykhailo_Zharzhailo | https://www.wikidata.org/wiki/Q20067054
Borys Humeniuk,M,1965,pre_1970,west_ukraine,west_ukraine,unknown,Ukrainian,,,,born: Ostriv village Ternopil Oblast (1965); ATO volunteer 2014; re-enlisted Feb 2022; MIA since Dec 2022; region_freeze unknown
# sources: https://euromaidanpress.com/2026/02/27/poet-went-to-war-twice-and-vanished-his-friend-published-the-book-he-left-behind/ | https://day.kyiv.ua/en/article/culture/poems-war
Elizaveta Zharikova,F,1996,1990s,east_ukraine,kyiv,unknown,Ukrainian,,,,born: Novopskov Luhansk Oblast (Liza); ~1996 per academic/scholarship profiles (exact DOB not public; some sources say 1989); piano Sievierodonetsk; MA Ukrainian philology Shevchenko Kyiv Univ; Maidan 2014; TDF combat medic since Feb 2022
# sources: https://book.artarsenal.in.ua/en/guest-2024/yelyzaveta-zharikova-2/ | https://chytomo.com/en/poet-and-soldier-liza-zharikova-ideally-my-collection-should-be-presented-while-i-m-still-alive/ | https://www.versopolis.com/poet/436/yelyzaveta-zharikova | https://uk.wikipedia.org/wiki/%D0%96%D0%B0%D1%80%D1%96%D0%BA%D0%BE%D0%B2%D0%B0_%D0%84%D0%BB%D0%B8%D0%B7%D0%B0%D0%B2%D0%B5%D1%82%D0%B0_%D0%92%D0%BE%D0%BB%D0%BE%D0%B4%D0%B8%D0%BC%D0%B8%D1%80%D1%96%D0%B2%D0%BD%D0%B0 | https://ui.org.ua/en/artists/yelyzaveta-zharikova-2/
Dmytro Lazutkin,M,1978,1970s,kyiv,kyiv,kyiv,Ukrainian,,,,born: Kyiv (18 Nov 1978); PEN: PEN member; mobilized summer 2023 47th Brigade Magura; MoD spokesperson since 2024; Shevchenko Prize 2024
# sources: https://en.wikipedia.org/wiki/Dmytro_Lazutkin | https://pen.org.ua/en/members/lazutkin-dmytro | https://kyivindependent.com/popular-poet-named-defense-ministry-spokesperson/ | https://ukraineworld.org/en/articles/stories/story-148
Eva Tur,F,1987,1980s,central_ukraine,unknown,unknown,Ukrainian,,,,born: Poltava region (15 Nov 1987); poet· artist· military servicewoman; ZSU since 2014 (paramedic; drone pilot); verses on war and Ukraine history; widow of fallen fighter Maksym Zapichnyi; Jan 2022/current location unknown
# sources: https://book.artarsenal.in.ua/en/guest-2025/eva-tur/ | https://www.londonukrainianreview.org/posts/poetry-by-ukraines-defenders | https://www.weareukraine.info/special/the-story-of-military-servicewoman-and-artist-eva-tur/ | https://euromaidanpress.com/2024/06/10/war-poetry-ukrainian-female-fighter-pens-powerful-words-on-gender-identity/
Iryna Vikyrchak,F,1988,1980s,west_ukraine,unknown,unknown,Ukrainian,,,,born: Zalishchyky Ternopil Oblast (17 May 1988); co-founded Intermezzo festival; 2024 bilingual collection Letters from Kolkata; Jan 2022/current unknown (hard-stop)
# sources: https://www.theemptysquare.org/the-participants/iryna-vikyrchak | https://www.wikidata.org/wiki/Q16695745 | https://losthorsepress.org/catalog/algometry-poems-by-iryna-vikyrchak-translated-from-the-ukrainian-by-nina-murray/
Olena Boryshpolets,F,1980,1980s,south_ukraine,south_ukraine,diaspora,Ukrainian,,,,born: Odesa (4 Apr 1980 per promegalit); poet writer journalist actress culture manager; Paustovsky Prize Blue Star; co-founder Creation Without Borders; post-Feb 2022 Poland (play Life in the Event of War); writer-in-residence City of Asylum + Research Scholar Pitt GSC
# sources: https://cityofasylum.org/artist/olena-boryshpolets/ | https://www.global.pitt.edu/gsc/faculty/pittsburgh-network-threatened-scholars | https://cityofasylum.org/program/olena-boryshpolets/ | https://www.promegalit.ru/personals/448_borishpolets_elena.html
Kateryna Babkina,F,1985,1980s,west_ukraine,kyiv,kyiv,Ukrainian,,,,born: Ivano-Frankivsk (22 July 1985); Jan 2022: kyiv; current: kyiv (may shuttle Kyiv/Warsaw); 2021 Angelus Award; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Kateryna_Babkina | https://pen.org.ua/en/members/babkina-kateryna | https://www.meridiancz.com/en/kateryna-babkina-ukraine/
Danik Zadorozhnyi,M,1995,1990s,west_ukraine,west_ukraine,west_ukraine,bilingual,,,,born: Lviv 1995 (also Danyil); bilingual family UA/RU; journalist; Dangerous Forms of Intimacy (2021); A. Dragomoshchenko Prize 2019; Words Without Borders contributor
# sources: https://wordswithoutborders.org/contributors/view/danyil-zadorozhnyi/ | https://lareviewofbooks.org/contributor/danyil-zadorozhnyi/ | https://www.dki.lv
Ludmila Khersonskaya,F,1964,pre_1970,born_abroad,south_ukraine,diaspora,Russian,,,,born: Tiraspol Moldavian SSR (born_abroad); long-time Odesa with Boris Khersonsky; as of 2023 Europe residency (Civitella Ranieri etc.) → diaspora at freeze
# sources: https://www.wordsforwar.com/lyudmyla-khersonska-bio | https://lannan.georgetown.edu/lyudmyla-khersonska/ | https://civitella.org/fellow/ludmila-khersonsky/ | https://www.arrowsmithpress.com/lyudmyla-khersonska
Hryhoryi Falkovych,M,1940,pre_1970,kyiv,kyiv,west_ukraine,bilingual,,,,born: Kyiv (25 June 1940); pre-invasion Kyiv; evacuated Kolomyia (west_ukraine) during 2022 war; Jewish-Ukrainian poet; initially Russian; now also Ukrainian → bilingual; PEN: PEN member
# sources: https://pen.org.ua/en/members/falkovych-grygorij | https://book.artarsenal.in.ua/en/guest-2025/hryhorii-falkovych/ | https://ukrainianjewishencounter.org/en/nash-holos-hryhorii-falkovych/ | https://theconversation.com/ukraine-war-a-wave-of-books-to-give-traumatised-children-hope-231750
Ostap Slyvynsky,M,1978,1970s,west_ukraine,west_ukraine,west_ukraine,Ukrainian,,,,born: Lviv (14 Oct 1978); Jan 2022: west_ukraine (Lviv); current: west_ukraine (Lviv); teaches Polish lit Ivan Franko Univ; VP PEN Ukraine; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Ostap_Slyvynsky | https://www.lyrikline.org/en/authors/ostap-slyvynsky | https://literaturfestival.com/en/authors/ostap-slyvynsky/
Oleh Kotsarev,M,1981,1980s,east_ukraine,kyiv,kyiv,Ukrainian,,,,born: Kharkiv (22 Aug 1981); Jan 2022: kyiv; current: kyiv (Bucha Kyiv Oblast per NYBooks 2023); PEN: PEN member
# sources: https://chytomo.com/en/authors/oleh-kotsarev-en/ | https://www.nybooks.com/contributors/oleh-kotsarev/ | https://www.meridiancz.com/en/oleh-kotsarev/
Pavlo Korobchuk,M,1984,1980s,west_ukraine,west_ukraine,west_ukraine,Ukrainian,,,,born: Lutsk (12 Jul 1984); poet writer musician journalist; son of poet Petro Korobchuk; PEN: PEN member; collections incl. Around the Clock! (2007) with Oleh Kotsarev
# sources: https://pen.org.ua/en/members/korobchuk-pavlo | https://www.lyrikline.org/en/authors/pawlo-korobtschuk | https://ukraineworld.org/en/articles/stories/writer-serviceman-pavlo-korobchuk | https://uk.wikipedia.org/wiki/%D0%9F%D0%B0%D0%B2%D0%BB%D0%BE_%D0%9A%D0%BE%D1%80%D0%BE%D0%B1%D1%87%D1%83%D0%BA
Maria Galina,F,1958,pre_1970,born_abroad,south_ukraine,south_ukraine,Russian,,,,born: Kalinin/Tver Russia (born_abroad); grew up Kyiv then Odesa; left Moscow Jan 2022 for Odesa; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Maria_Galina | https://pen.org.ua/en/members/galina-mariya | https://publishingperspectives.com/2022/03/ukraine-maria-galina-on-the-russian-language-as-a-trigger/
Ivan Andrusiak,M,1968,pre_1970,west_ukraine,kyiv,kyiv,Ukrainian,,,,born: Verbovets Kosiv district Ivano-Frankivsk Oblast (28 Dec 1968); Jan 2022: kyiv; current: kyiv (Berezan Kyiv Oblast); Fontan Kazok publisher editor; PEN: PEN member
# sources: https://pen.org.ua/en/members/andrusyak-ivan-myhajlovych | https://www.ibby.org/archive-storage/12_HCAA_Dossiers/2020_Authors/Andrusyak_2.pdf | http://fontan-book.com/en/authors-en/ivan-andrusyak
Oles Ilchenko,M,1957,pre_1970,kyiv,diaspora,diaspora,Ukrainian,,,,born: Kyiv (4 Oct 1957); writer painter screenwriter; City with Chimeras Best Ukrainian Book 2009; Geneva since 2011; 35 books; TEFI Teletriumph; PEN: PEN member; poems translated EN DE FR SR LT RU
# sources: https://pen.org.ua/autors/ilchenko-oles | https://olesilchenko.com/about | https://www.wikidata.org/wiki/Q4199974
Yaryna Chornohuz,F,1995,1990s,kyiv,kyiv,east_ukraine,Ukrainian,,,,born: Kyiv (1995); Jan 2022: kyiv; current: east_ukraine (frontline Donbas/Kherson); 140th Marine Recon combat medic/drone op since 2019; 2024 Shevchenko Prize; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Yaryna_Chornohuz | https://kyivindependent.com/in-every-woman-there-is-a-soldier-yaryna-chornohuz-the-brave-poet-fighting-russia/ | https://www.arrowsmithpress.com/journal/featured-poets-marchant-chornohuz
Artur Dron,M,2000,2000s_plus,west_ukraine,west_ukraine,east_ukraine,Ukrainian,,,,born: Vaskrasyntsi Ivano-Frankivsk Oblast (31 Dec 2000); Lviv; LNU Franko journalism; Stary Lev event manager; 125th TRO Brigade since 2022; Donetsk Oblast combat from Aug 2022; collections Gurtozhitok No. 6 Here We Were (Tut buly my)
# sources: https://taubinpoetry.com/en/authors-en/arthur-dron/ | https://starylev.com.ua/news/50-virsiv-z-viyny-pro-lyudey-na-viyni-vyxodyt-drukom-zbirka-tut-buly-my-poeta-viyskovosluzbovcya-artura-dronya | https://book.artarsenal.in.ua/en/guest-2023/artur-dron-2/ | https://galka.if.ua/nikhto-z-nas-z-tsym-bolem-ne-sam-istoriia-23-richnoho-viyskovosluzhbovtsia-y-poeta-artura-dronia-foto/
Vasyl Makhno,M,1964,pre_1970,west_ukraine,diaspora,diaspora,Ukrainian,,,,born: Chortkiv Ternopil Oblast (8 Oct 1964); in NYC since 2000 → diaspora; PEN: PEN member; other locations: New York
# sources: https://en.wikipedia.org/wiki/Vasyl_Makhno | https://www.wordsforwar.com/vasyl-makhno-bio | https://www.fekt.org/vasyl-makhno/
Alie Kenzhalieva,F,1986,1980s,born_abroad,crimea,crimea,bilingual,,,,born: Uzbekistan (2 May 1986; aka Aliye Kendzhe-Ali); family returned Crimea 1988; Crimean Tatar poet/writer; Ozen near Alushta; writes RU+Crimean Tatar; Feza (2022); novel Cradle; 2018+ persecution occupied Crimea; CTRC/PEN coverage
# sources: https://uk.wikipedia.org/wiki/%D0%90%D0%BB%D1%96%D1%94_%D0%9A%D0%B5%D0%BD%D0%B4%D0%B6%D0%B0%D0%BB%D1%96%D1%94%D0%B2%D0%B0 | https://ctrcenter.org/en | https://pen.org.ua/en/crimean-tatar-poet-under-criminal-investigation-in-russian-occupied-crimea-for-anti-war-verse | http://rozstaje.art/en/autorzy/alije-kendze-ali/
Arkadii Shtypel,M,1944,pre_1970,born_abroad,south_ukraine,south_ukraine,bilingual,,,,born: Kattakurgan Uzbekistan (14 Mar 1944; WWII evacuation); childhood/youth Dnipro; bilingual poet translator; Dnipropetrovsk Univ physics then Moscow; returned Odesa with wife Maria Galina; died Odesa 2024-10-23 age 80; PEN: PEN member
# sources: https://pen.org.ua/in-memoriam/shtypel-arkadij | https://pen.org.ua/pomer-poet-ta-perekladach-arkadij-shtypel | https://eastwestliteraryforum.com/poetry/arkady-shtypel-two-poems-translated-by-maria-bloshteyn/
Yanis Sinayko,unknown,,unknown,unknown,unknown,unknown,unknown,,,,PENDING: no reliable source found in batch; recommend Research-mode follow-up
# sources: (none — searched: https://ukrpoetry.org/)
Natalia Belczenko,F,,unknown,kyiv,kyiv,unknown,bilingual,,,,Kyiv-based; shifted Russian to Ukrainian per Glaser; Lviv Book Forum prize ~2022; birth year and current location not confirmed in batch
# sources: https://associationforjewishstudies.org/contemporary-in-justice-more-than-war-songs-what-ukraine-s-poets-teach-us-about-language-and-community
Maksym Kryvtsov,M,1990,1990s,west_ukraine,west_ukraine,east_ukraine,Ukrainian,,,,born: Rivne (22 Jan 1990); KIA 7 Jan 2024 Kharkiv region; Hero of Ukraine posthumous 2025; volunteered Right Sector 2014; rejoined Feb 2022
# sources: https://en.wikipedia.org/wiki/Maksym_Kryvtsov | https://kyivindependent.com/ukrainian-poet-maksym-kryvtsov-killed-on-front-line/ | https://war.ukraine.ua/heroes/ukrainian-poet-maksym-kryvtsov-was-killed-in-action-on-january-7/
Anatoliy Dnistrovyi,unknown,,unknown,unknown,unknown,unknown,unknown,Ukrainian,,,,Ukrainian poet/critic; birth year/place not confirmed in batch; incidental lit-criticism sources only
# sources: https://news.exeter.ac.uk/faculty-of-humanities-arts-and-social-sciences/uk-ukraine-partnership-brings-wartime-poetry-to-the-world/
Олександр Ірванець,M,1961,pre_1970,west_ukraine,kyiv,kyiv,Ukrainian,,,,Cyrillic alias of Oleksandr Irvanets (row 4); identical biography; born: Lviv; raised Rivne; Irpin; Bu-Ba-Bu co-founder; PEN: PEN member
# sources: https://en.wikipedia.org/wiki/Oleksandr_Irvanets | https://pen.org.ua/en/members/irvanets-oleksandr | https://www.wikidata.org/wiki/Q3489996
Dasha Suzdalova,unknown,,unknown,unknown,unknown,unknown,unknown,,,,PENDING: no reliable source found in batch; recommend Research-mode follow-up
# sources: (none — searched: https://ukrpoetry.org/)
"""


def main() -> None:
    entries = parse_batch(BATCH)
    patches = {e[0]["author"]: e[0] for e in entries}
    raw_lines = {e[0]["author"]: e[1] for e in entries}

    text = COV.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    hdr = next(csv.reader([lines[0]]))
    rows: list[dict[str, str]] = []
    changed: list[str] = []
    missing: list[str] = []

    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(hdr, next(csv.reader([ln]))))
        author = r["author"]
        if author not in patches:
            rows.append(r)
            continue
        p = patches[author]
        for field in RESEARCH_FIELDS:
            if field in p:
                r[field] = p[field]
        # Corpus: only set if batch provided non-empty values
        for field in CORPUS_FIELDS:
            if p.get(field):
                r[field] = p[field]
        before = ln
        # rebuild not needed for diff; compare key fields
        key_vals = {f: r.get(f, "") for f in RESEARCH_FIELDS}
        patch_vals = {f: p.get(f, "") for f in RESEARCH_FIELDS}
        if key_vals != patch_vals:
            changed.append(author)
        rows.append(r)
        del patches[author]

    missing = list(patches.keys())
    if missing:
        raise SystemExit(f"Authors not in CSV: {missing}")

    with COV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    with AUDIT.open(encoding="utf-8-sig", newline="") as f:
        fn = list(csv.DictReader(f).fieldnames)
        audit = list(csv.DictReader(open(AUDIT, encoding="utf-8-sig")))

    by_author_meta: dict[str, dict] = {}
    for a in audit:
        by_author_meta.setdefault(a["author"], a)

    updated_audit = 0
    for entry, _ in entries:
        author = entry["author"]
        meta = by_author_meta.get(author)
        if not meta:
            continue
        research = entry["notes"][:400] if entry.get("notes") else f"PENDING batch for {author}"
        sources = entry.get("references", "").replace(" | ", "; ")
        raw = f"{research} | sources: {sources}"[:800] if sources else research[:800]

        for field in RESEARCH_FIELDS:
            val = entry.get(field, "")
            found = False
            for a in audit:
                if a["author"] == author and a["field"] == field:
                    a["filled_value"] = val
                    a["provenance"] = PROV
                    a["source_table"] = TABLE
                    a["source_column"] = "external_web_research"
                    a["source_raw_value"] = raw
                    found = True
                    updated_audit += 1
            if not found:
                audit.append(
                    {
                        "cov_row": meta["cov_row"],
                        "author": author,
                        "field": field,
                        "filled_value": val,
                        "provenance": PROV,
                        "xlsx_row_idx": meta["xlsx_row_idx"],
                        "xlsx_author": meta["xlsx_author"],
                        "source_table": TABLE,
                        "source_column": "external_web_research",
                        "source_raw_value": raw,
                    }
                )
                updated_audit += 1

    with AUDIT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(audit)

    print(f"Applied {len(entries)} authors")
    if changed:
        print(f"CSV field changes: {', '.join(changed)}")
    else:
        print("CSV already matched batch (audit refreshed)")


if __name__ == "__main__":
    main()
