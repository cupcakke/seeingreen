Biztonság

Érzékeny adatok

A promptok, válaszok, hivatkozások, alcsoportmetaadatok és modellválaszok tartalmazhatnak személyes, bizalmas, szabályozott vagy védett információkat. A strukturált naplók az engedélyezett maszkolási minták alapján redaktálnak. Az auditanyagok nem redaktálnak forrásadatokat, mert a teljes rekonstruálhatóság szükséges. Az artefaktumgyökérkönyvtárat és az adatbázist érzékeny adattárolóként kell védeni.

Hitelesítés

Az HTTP API a kulcsokat az X-API-Key fejlécen keresztül fogadja. A kulcsokat az EUQ_API_KEYS-ben konfigurált készlettel hasonlítja össze. TLS mögé kell telepíteni. Véletlenszerű, kellő entrópiájú kulcsokat kell használni, és átfedő kulcskészleteken keresztül kell őket forgatni.

Sebességkorlátozás

A Redis minden API-kulcshoz elosztott, fix ablakos korlátozást biztosít. Ha a Redis nem elérhető, minden folyamat memóriabeli gördülő ablakot használ. Több replikás telepítések esetén figyelni kell a Redis készenléti állapotát, mivel a helyi tartalék nagyobb összesített sebességet tesz lehetővé.

Fájlrendszer-hozzáférés

A kötegelt kiértékelés feloldja az adathalmaz és az opcionális konfigurációs útvonalakat, és megköveteli, hogy azok az EUQ_ALLOWED_DATA_ROOT leszármazottai legyenek. A szimbolikus linkek feloldása a tartalmazási ellenőrzés előtt történik. Az alfolyamat háttérprogramparancsa rendszergazdai konfiguráció, és soha nem lehet írható megbízhatatlan API-kliensek számára.

Háttérprogram-hitelesítő adatok

A HTTP háttérprogram hitelesítő adatai az api_key_env által megadott környezeti változóból töltődnek be. A hitelesítő adatok soha nem szerepelnek a sorosított háttérprogram-konfigurációban vagy a kérés-reprodukálhatósági metaadatokban. A háttérprogram-fájlokban közvetlenül konfigurált további fejlécek sorosításra kerülnek, ezért nem tartalmazhatnak titkokat.

Függőség- és képmenedzsment

Az építési képeket ellenőrzött függőségi manifesztumokból kell előállítani. Vizsgálja át a Python-függőségeket és a konténerképeket. Rögzítse az éles képek digestjeit a telepítési infrastruktúrában. Biztonsági frissítések után építse újra, és futtassa le ismét a teljes teszt- és kalibrációs kiértékelési csomagot.

Sebezhetőségek jelentése

A sebezhetőségeket bizalmasan jelentse a telepítés tulajdonosának az érintett verzióval, konfigurációval, reprodukálási lépésekkel és hatással együtt. A jelentés ne tartalmazzon élő hitelesítő adatokat, privát promptokat vagy nem redaktált felhasználói adatokat.
