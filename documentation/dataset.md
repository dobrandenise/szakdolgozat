# 4.2 Az adathalmaz szerkezete és időbeli felépítése

A dolgozat a BankSim szimulátor által előállított adathalmazt használja [8]. Az adathalmaz 594 643 tranzakciót tartalmaz, amelyek közül 587 443 jogszerű és 7200 csalárd, így a csalások aránya 1,21% [8]. A táblázat az adathalmaz attribútumait foglalja össze.

| Attribútum | Típus | Leírás |
|------------|--------|---------|
| step | numerikus | A szimulált nap sorszáma |
| customer | kategorikus | Ügyfélazonosító |
| age | kategorikus | Korcsoport |
| gender | kategorikus | Nem |
| zipcodeOri | kategorikus | Ügyfél irányítószáma |
| zipMerchant | kategorikus | Kereskedő irányítószáma |
| merchant | kategorikus | Kereskedőazonosító |
| category | kategorikus | Vásárlási kategória |
| amount | numerikus | Tranzakció összege (euró) |
| fraud | bináris | Csalás címkéje (0 = jogszerű, 1 = csalárd) |

A *step* attribútum az időrendet rögzíti, ahol egy érték egy szimulált napnak felel meg. Az adathalmaz időrendben rendezett, ami lehetővé teszi időfüggő jellemzők képzését és a valós működést jobban közelítő kiértékelési eljárások alkalmazását.

Az előfeldolgozás során a konstans attribútumok eltávolításra kerülnek, így a további elemzés és modellezés a releváns változókra épül.
