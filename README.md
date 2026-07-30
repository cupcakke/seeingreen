Epistemikus UQ

Az Epistemic UQ egy teljes bizonytalanság-kvantifikációs, magabiztosság-kalibrációs, szelektív-előrejelzési és audit rendszer nyelvi modell kimeneteihez. Célja annak becslése, hogy egy előállított válasz helyes-e, a modellismereti bizonytalanság elkülönítése a promptérzékenységtől és a dekódolási instabilitástól, a magabiztosság kalibrálása visszatartott helyességi címkékhez, valamint a kalibrált kockázat explicit downstream műveletekké alakítása.

A rendszer támogatja a fekete doboz OpenAI-kompatibilis HTTP API-kat, az Ollamát, a helyi Hugging Face kauzális nyelvi modelleket, valamint bármely helyi végrehajtható programot, amely megvalósítja a mellékelt JSON protokollt. A promptokat, kéréseket, generálásokat, tokenvalószínűségeket, magabiztossági lekérdezéseket, igazságellenőrzési válaszokat, sztochasztikus mintákat, promptperturbációkat, szemantikus klasztereket, jellemzőket, címkéket, szabályzati döntéseket, költségeket, késleltetést és reprodukálhatósági metaadatokat tartalomcímzett auditartifaktumokban őrzi meg.

Megvalósított bizonytalansági jelek

Az alapválasz a következő független jelekkel kerül kiértékelésre, amikor a kiválasztott háttérrendszer támogatja őket:

1. Numerikus önbevallott magabiztosság, amelyet egy korlátozott JSON magabiztossági lekérdezésből nyerünk.
2. Válaszszintű magabiztosság, amelyet a token logvalószínűségekből aggregálunk geometriai közép, számtani közép, minimum, szorzat vagy hossz-normalizált szorzat segítségével.
3. Igazságellenőrzési valószínűség egy korlátozott helyes-versus-helytelen lekérdezésből.
4. Önkonzisztencia ismételt sztochasztikus generálásokból, rögzített prompt mellett.
5. Promptperturbációs stabilitás determinisztikus utasítássorrend-, formázás-, feladatkeret- és lexikai transzformációk mellett.
6. Különböző modellek közötti egyezés normalizált alapválaszokon.
7. Szemantikus egyezés, domináns klaszttömeg, szemantikus entrópia, lexikai egyezés és ellentmondásdetektálás.
8. Kompozit episztemikus kockázat, amely modellismereti bizonytalanságra, promptérzékenységi bizonytalanságra és dekódolási-instabilitási bizonytalanságra bomlik.
9. Átlátható tanult fúzió vagy determinisztikus szabályalapú fúzió.
10. Utólagos kalibráció hőmérsékleti skálázással, Platt-skálázással, izotonikus regresszióval vagy béta-kalibrációval.

Architektúra

A forrásfa explicit rétegekre van osztva:

* A schemas.py tartalmazza az immutábilis Pydantic sémákat a promptokhoz, kérésekhez, generálásokhoz, tokenvalószínűségekhez, mintákhoz, válaszokhoz, klaszterekhez, perturbációkhoz, kalibrációs bin-ekhez, címkékhez, alcsoport-auditokhoz, döntésekhez, manifesztumokhoz, drift-pillanatképekhez és kísérletfuttatásokhoz.
* A backends tartalmazza az OpenAI-kompatibilis HTTP, Ollama, Hugging Face és subprocess JSON protokoll háttérrendszerek gyártási adaptereit.
* A processing tartalmazza az adathalmaz-normalizálást, a válaszkanonikalizálást, a determinisztikus perturbációgenerálást, valamint a pontos, regexes, numerikus, token-F1 és strukturált validátorokat.
* Az uncertainty tartalmazza az önbevallás-elemzést, a logvalószínűség-aggregálást, a szemantikus elbírálást, a klaszterezést, az egyezési statisztikákat, az ellentmondásdetektálást és az episztemikus kockázatbontást.
* A calibration tartalmazza a megbízhatósági metrikákat, a szelektív-előrejelzési metrikákat, négy felügyelt kalibrátort, az átlátható monoton fúziót, a determinisztikus fúziót, az alcsoport-auditokat, a grouping-loss proxykat és a legrosszabb szelet felderítését.
* A policy.py a kalibrált magabiztosságot és a kockázati korrekciókat válasz-, figyelmeztetés-, pontosítás-, ellenőrzés-, tartózkodás- vagy emberi eszkalációs műveletekké alakítja.
* A storage.py relációs kísérletmetaadatokat és tartalomcímzett, tömörített artifaktum-megőrzést tartalmaz.
* A runner.py teljes kísérleteket futtat, párhuzamosítja a példákat, folytatja a befejezett munkát, megőrzi minden köztes artifaktumot, és rögzíti a driftmegfigyeléseket.
* A reports.py gépileg olvasható JSON-t, emberileg olvasható HTML-t, megbízhatósági diagramokat, magabiztossági hisztogramokat, lefedettség–pontosság görbéket, legrosszabb szelet táblázatokat, ellentmondásos eseteket és túlzottan magabiztos hibákat generál.
* A service.py hitelesített, rate-limitált HTTP API-kat, Prometheus-metrikákat, állapotellenőrzéseket, strukturált naplókat és trace-azonosítókat tesz elérhetővé.
* A cli.py adatállomány-, háttérrendszer-, kísérlet-, metrika-, jelentés-, küszöb-, kalibráció- és fúzióparancsokat tesz elérhetővé.

Telepítés

Python 3.11 vagy újabb szükséges.

python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e . --no-deps
cp .env.example .env
alembic upgrade head

Hugging Face-hez és embedding-alapú szemantikus ekvivalenciához:

python -m pip install torch transformers sentence-transformers

Teszteléshez:

python -m pip install -r requirements-test.txt
python -m pip install --no-build-isolation -e . --no-deps
pytest

Azonnali, végponttól végpontig tartó ellenőrzés

A repozitórium tartalmaz egy determinisztikus helyi referencia motort, amely ugyanazt a subprocess protokollt valósítja meg, mint egy helyi modellfolyamat. Valódi aritmetikai, összehasonlítási, hangulatelemzési, kinyerési és strukturált kimeneti feladatokat végez, és illesztett tokenvalószínűségeket ad vissza. Az end-to-end parancs lefuttatja az alapgenerálást, az önbevallást, az igazságellenőrzést, a sztochasztikus mintavételezést, a promptperturbációkat, az értékelést, a perzisztálást, a kalibrációs metrikákat, a szabályzati következtetést és a jelentéskivonatot.

cp .env.example .env
python scripts/bootstrap.py
python scripts/e2e.py

A kísérletazonosító JSON-ként kerül kiírásra. A jelentések a var/reports/<experiment_id>/report.html és a var/reports/<experiment_id>/report.json útvonalak alá íródnak.

Háttérrendszer-regisztráció

A helyi referencia háttérrendszer regisztrálása:

epistemic-uq backend register --path examples/backend-subprocess.json

Ollama regisztrálása a modell helyi telepítése után:

ollama pull llama3.1:8b
epistemic-uq backend register --path examples/backend-ollama.json

Hugging Face modell regisztrálása:

epistemic-uq backend register --path examples/backend-huggingface.json

OpenAI-kompatibilis szerver regisztrálása, amely a http://localhost:8080/v1/chat/completions címen fut:

epistemic-uq backend register --path examples/backend-http.json

A HTTP adapter chat-completion kéréseket küld hőmérséklettel, top-p-vel, maximális tokenekkel, opcionális maggal és opcionális logvalószínűségi mezőkkel. A szolgáltató-specifikus fejlécek és payload mezők a háttérkonfiguráció options.headers és options.payload leképezésein keresztül adhatók meg. A titkos értékek a környezetből töltődnek be, amikor az api_key_env konfigurálva van.

Adatállomány-formátum

JSON Lines, JSON tömbök, examples tömböt tartalmazó JSON objektumok és CSV elfogadott. Minden normalizált rekord a következő mezőket tartalmazza:

{
  "example_id": "math-001",
  "dataset_id": "reference-benchmark",
  "task_type": "question_answering",
  "user_input": "What is 17 + 25?",
  "expected_format": "number",
  "reference_label": 42,
  "valid_answers": [],
  "subgroup_metadata": {
    "domain": "arithmetic",
    "difficulty": "easy"
  },
  "perturbation_rules": {},
  "validator_config": {
    "method": "numeric"
  },
  "criticality": "low",
  "metadata": {}
}

Érvényes feladattípusok: question_answering, classification, extraction és structured. Érvényes validátor-módszerek: auto, regex, numeric, structured és token_f1. A numerikus validátorok elfogadják az absolute_tolerance és relative_tolerance értékeket. A strukturált validátorok elfogadják a required_keys értékeket. A Token-F1 validátorok elfogadják a threshold értéket.

Kísérletvégrehajtás

epistemic-uq dataset register \
  --path examples/dataset.jsonl \
  --dataset-id reference-benchmark \
  --version 1
epistemic-uq experiment run \
  --dataset examples/dataset.jsonl \
  --dataset-id reference-benchmark \
  --backend reference-local \
  --config config/e2e.yaml

Több --backend opció keresztmodell-egyezést tesz lehetővé:

epistemic-uq experiment run \
  --dataset data/evaluation.jsonl \
  --dataset-id evaluation-v1 \
  --backend ollama-local \
  --backend smollm2-local \
  --config config/default.yaml

A megszakított kísérletek azonosító alapján folytathatók anélkül, hogy az elkészült példá–háttérrendszer párokat újrafuttatnánk:

epistemic-uq experiment run \
  --dataset data/evaluation.jsonl \
  --dataset-id evaluation-v1 \
  --backend ollama-local \
  --resume "$EXPERIMENT_ID"

A folytatás-ellenőrzés elutasítja a megváltozott adatállomány-hash-eket vagy a megváltozott konfiguráció-hash-eket.

Metrikák és jelentések

epistemic-uq metrics compute \
  --experiment-id "$EXPERIMENT_ID" \
  --source calibrated_confidence \
  --bins 15 \
  --strategy quantile
epistemic-uq report generate \
  --experiment-id "$EXPERIMENT_ID" \
  --output-dir "var/reports/$EXPERIMENT_ID"

A metrikák közé tartozik a pontosság, a megbízhatósági bin-ek, az elvárt kalibrációs hiba, a maximális kalibrációs hiba, a Brier-pontszám, a negatív logvalószínűség, az AUROC, amikor mindkét osztály jelen van, a kockázat–lefedettség görbe alatti terület, a lefedettség, a pontosság, a kockázat és a tartózkodási darabszámok.

Felügyelt kalibráció

A kalibrációt csak egy fejlesztési kísérleten illesztjük, és egy külön, visszatartott tesztkísérleten értékeljük:

epistemic-uq calibration fit \
  --development-experiment "$DEVELOPMENT_EXPERIMENT_ID" \
  --test-experiment "$TEST_EXPERIMENT_ID" \
  --source self_consistency_confidence \
  --method isotonic \
  --calibrator-id self-consistency-isotonic-v1

A tárolt artifaktum tartalmazza az illesztett paramétereket, a fejlesztési metrikákat kalibráció előtt és után, a tesztmetrikákat kalibráció előtt és után, a forrásazonosságot, a kísérletazonosítókat, az illesztési időbélyeget és egy tréningadat-hash-t. A kalibrátort egy pipeline-konfigurációhoz így lehet hozzáadni:

calibration:
  calibrator_id: self-consistency-isotonic-v1

Átlátható magabiztosság-fúzió

A fúzió szigorú train, validation és test kísérleti szeparációt használ. A validation kísérlet választja ki az L2 regularizációs erősséget. A végső átlátható modell a train plus validation adatokon kerül illesztésre csak a modellválasztás után, a test kísérlet pedig a végső értékelésig érintetlen marad.

epistemic-uq fusion fit \
  --train-experiment "$TRAIN_EXPERIMENT_ID" \
  --validation-experiment "$VALIDATION_EXPERIMENT_ID" \
  --test-experiment "$TEST_EXPERIMENT_ID" \
  --model-id transparent-fusion-v1 \
  --monotonic

A modellt a konfigurációhoz így lehet hozzáadni:

fusion:
  mode: learned
  model_id: transparent-fusion-v1

A modell exportálja a jellemzőneveket, a nemnegatív monoton koefficienseket, az interceptet, a normalizációs statisztikákat, a hiányzóérték-imputációs értékeket, a kiválasztott regularizációt, a train metrikákat, a validation metrikákat, a test metrikákat és a tréning-hash-eket.

Küszöbszimuláció

epistemic-uq threshold simulate \
  --experiment-id "$EXPERIMENT_ID" \
  --source calibrated_confidence \
  --objective minimum_risk \
  --max-risk 0.05

A támogatott célfüggvények: utility, minimum_risk és target_coverage. A hasznosságoptimalizálás elfogadott, hibás és tartózkodási hasznosságokat vesz figyelembe. A minimumkockázat-optimalizálás a legnagyobb lefedettségű megvalósítható küszöböt adja vissza. A cél-lefedettség optimalizálás a kért lefedettséghez legközelebb eső működési pontot adja vissza.

Szolgáltatás API

A szolgáltatás indítása:

uvicorn epistemic_uq.service:app --host 0.0.0.0 --port 8000

Az alapértelmezett helyi API kulcs a .env.example fájlból: local-development-key.

Állapot és metrikák:

curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/metrics

Egyetlen lekérdezéses bizonytalanságbecslés:

curl -sS http://localhost:8000/v1/uncertainty \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-development-key' \
  -d '{
    "backend_ids": ["reference-local"],
    "task": {
      "example_id": "api-math-1",
      "dataset_id": "api",
      "task_type": "question_answering",
      "user_input": "What is 21 + 21?",
      "expected_format": "number",
      "reference_label": 42,
      "valid_answers": [],
      "subgroup_metadata": {"domain": "arithmetic"},
      "perturbation_rules": {},
      "validator_config": {"method": "numeric"},
      "criticality": "low",
      "metadata": {}
    },
    "generation": {},
    "config_overrides": {}
  }'

A válasz tartalmazza a normalizált választ, a teljes alapgenerálást, a nyers bizonytalansági jeleket, az episztemikus bontást, a kalibrált magabiztosságot, a hivatkozás esetén az értékelési címkét, a szabályzati műveletet, az okokat és az auditartifaktum elérési útját.

Szabályzati következtetés:

curl -sS http://localhost:8000/v1/policy/decide \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-development-key' \
  -d '{
    "calibrated_confidence": 0.72,
    "features": {
      "contradiction": false,
      "model_knowledge_uncertainty": 0.2,
      "prompt_sensitivity_uncertainty": 0.1,
      "decoding_instability_uncertainty": 0.15,
      "epistemic_risk": 0.18,
      "raw": {}
    },
    "criticality": "medium",
    "subgroup_metadata": {},
    "subgroup_audits": []
  }'

Küszöbszimuláció:

curl -sS http://localhost:8000/v1/thresholds/simulate \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-development-key' \
  -d '{
    "confidences": [0.95, 0.8, 0.45, 0.2],
    "labels": [1, 1, 0, 0],
    "utilities": {"objective": "minimum_risk", "max_risk": 0.05}
  }'

A kötegelt értékelés csak akkor fogad el szerver-helyi adatállomány- és konfigurációs útvonalakat, ha mindkettő az EUQ_ALLOWED_DATA_ROOT belsejébe oldódik fel. Ez megakadályozza a tetszőleges fájlrendszer-olvasásokat.

Subprocess háttérrendszer protokoll

Az adapter minden kérésnél egyszer indítja el a konfigurált parancsot, egy JSON objektumot ír a standard bemenetre, és egy JSON objektumot olvas a standard kimenetről. A bemenet tartalmazza a kérésazonosítót, a modellt, a promptot, a hőmérsékletet, a top-p-t, a maximális tokeneket, a magot, a stop stringeket, a logvalószínűség-kérést és a metaadatokat. A kimenet megköveteli a text mezőt, és támogatja a model, finish_reason, token_probabilities, usage és reproducibility mezőket.

A tokenvalószínűségi rekordok tartalmazzák a token, logprob, opcionális probability, position, opcionális start_char és opcionális end_char mezőket. A karaktertartományok lehetővé teszik a válaszspecifikus tokenkiválasztást több részből álló kimeneteknél.

Perzisztencia és auditálhatóság

A relációs metaadatok SQLAlchemy-n keresztül SQLite-ban vagy PostgreSQL-ben tárolódnak. A nyers és származtatott artifaktumok kanonikus, rendezett JSON-ként kerülnek sorosításra, determinisztikus gzip metaadatokkal tömörítve, SHA-256-tal hashelve, atomikusan írva és tartalomhash alapján címezve. Minden eredmény egy auditartifaktumhoz kapcsolódik, amely tartalmazza:

* Az alap kérés és generálás.
* Az önbevallási generálás és a parszolt magabiztosság.
* Az igazságellenőrzési generálás és a parszolt valószínűség.
* Minden sztochasztikus minta és normalizált válasz.
* Minden szemantikus klaszter és egyezési statisztika.
* Minden perturbációs definíció, kérés, generálás, válasz és ekvivalenciaítélet.
* Keresztmodell alapválaszok és generálásazonosítók.
* Bizonytalansági jellemzők és bontás.
* Fúziós hozzájárulások és utókalibrációs valószínűség.
* Értékelési címke és szabályzati döntés.

A kísérletmanifesztumok megőrzik az adatállomány-hash-eket, a háttérrendszer-konfigurációkat, a modellazonosítókat, a prompt sablonazonosítókat és verziókat, a dekódolási paramétereket, a magokat, az időbélyegeket, a konfigurációt, a forrásutakat és a háttérrendszer reprodukálhatósági metaadatait.

Megfigyelhetőség

A strukturált JSON naplók trace-azonosítókat tartalmaznak. Minden HTTP válasz tartalmazza az X-Trace-ID és X-Response-Time-Ms fejléceket. A Prometheus-metrikák lefedik a modellhívásokat, háttérrendszer-hibákat, késleltetést, prompt tokeneket, befejezési tokeneket, parszolási hibákat, szabályzati műveleteket, a közelmúltbeli magabiztosságátlagokat és a driftriasztásokat.

A driftmonitor populációstabilitási indexet számol a referencia és a jelenlegi ablakok között, és opcionálisan összehasonlítja a magabiztosság–helyesség hibát. A driftmegfigyelések jelek, feladattípusok és alcsoportmetaadatok szerint vannak szeletelve, és akkor kerülnek perzisztálásra, amikor egy teljes ablak rendelkezésre áll.

A konfigurált redakció eltávolítja az e-mail címeket, telefonszámokat, hitelkártya-szerű szekvenciákat és opcionális IP-címeket a strukturált naplókból. A nyers auditartifaktumok azonban redaktálatlanok maradnak, mert ezek a forrásrekordok, és megfelelő fájlrendszeri, adatbázis-, titkosítási, megőrzési és hozzáférés-vezérlési szabályzatokkal kell védeni őket az adott telepítésnek megfelelően.

Telepítés

Helyi konténerek:

docker compose up --build

A Compose stack az API-t, a PostgreSQL 16-ot és a Redis 7-et futtatja. A PostgreSQL tárolja a kísérletmetaadatokat. A Redis biztosítja az elosztott, fix ablakos API rate limitinget. Az artifaktumadatok egy perzisztens kötetben tárolódnak.

Hálózati telepítés előtt:

1. Cseréld le a fejlesztői API kulcsot legalább egy kriptográfiailag véletlen kulcsra az EUQ_API_KEYS változóban.
2. Termináld a TLS-t egy megbízható reverse proxyban vagy service meshben.
3. Korlátozd az artifaktum- és adatbázishozzáférést az API identitására.
4. Konfiguráld a PostgreSQL mentéseket és az artifaktumkötet-pillanatképeket.
5. Állítsd be a promptokra és generálásokra vonatkozó megőrzési követelményeket.
6. Konfiguráld a logtovábbítást és a Prometheus scrapinget.
7. Futtesd a migrációkat az alembic upgrade head paranccsal az új alkalmazásverzió telepítése előtt.
8. Érvényesítsd a háttérrendszer-képességjelzőket a kiválasztott szolgáltatóval szemben.
9. Kalibrátorokat csak ugyanazon feladat- és háttérrendszer-feltételek mellett gyűjtött, címkézett fejlesztési adatokból illessz.
10. A modell-, prompt-, adatállomány- vagy szabályzatváltozások után újra auditáld az alcsoport-kalibrációt.

Tesztlefedettség

A tesztcsomag lefedi az immutábilis séma validálását, a háttérrendszer-konfiguráció validálását, a szöveges és strukturált normalizálást, a numerikus toleranciát, a magabiztosság-parszolást, az igazság-parszolást, a tokenvalószínűség-aggregálást, a szemantikus ekvivalenciát, a szemantikus klaszterezést, az ellentmondásdetektálást, a kalibrációs metrikákat, a szelektív görbéket, a kalibrátor-szerializálást, a monoton fúziót, a szabályfúziót, az alcsoport-auditokat, a legrosszabb szelet felderítését, a szabályzati műveleteket, az artifaktum-perzisztenciát, a relációs perzisztenciát, a subprocess protokoll végrehajtását, a redakciót, a driftdetektálást, a teljes egypéldányos végrehajtást, a teljes adathalmaz-végrehajtást, a folytatási viselkedést és a jelentésgenerálást. A Hypothesis tesztek validálják a szelektív-görbe valószínűségi és kockázat-invariánsait a generált magabiztossági tömbökön keresztül.
