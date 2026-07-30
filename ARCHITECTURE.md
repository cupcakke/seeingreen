Architektúra

Rendszercél

Minden nyelvi modell válaszához a rendszer egy kalibrált helyességi valószínűséget és egy episztemikus kockázati felbontást becsül. A felbontás elkülöníti a modell tudásához, a prompt érzékenységéhez és a dekódolási instabilitáshoz kapcsolódó bizonyítékokat. A kimenet olyan downstream irányelvekhez készült, amelyeknek el kell dönteniük, hogy válaszoljanak, figyelmeztessenek, pontosítást kérjenek, külső ellenőrzést kérjenek, tartózkodjanak a válaszadástól, vagy eszkaláljanak.

Adatfolyam

1. Egy adathalmazrekordot érvényesítenek egy megváltoztathatatlan kanonikus értékelési egységgé.
2. Egy alap promptot állítanak elő a feladatból és a várt válaszformátumból.
3. Minden konfigurált backend egy alap generálást készít log valószínűségekkel, amikor ez támogatott.
4. A válaszokat a feladattípus szerint kanonizálják.
5. A backend numerikus, önbevallott magabiztosságra és igazság-ellenőrzési valószínűségre vonatkozó lekérdezést kap.
6. Ismételt sztokasztikus minták generálódnak rögzített prompt mellett.
7. Determinisztikus, jelentésmegőrző prompt-perturbációk generálódnak és kiértékelődnek.
8. Több backend alapválaszait összehasonlítják.
9. A válaszokat lexikai, numerikus, címke- és opcionális beágyazási ekvivalencia szempontjából elbírálják.
10. Szemantikus klasztereket, domináns tömeget, entrópiát, egyezést és ellentmondásokat számítanak.
11. A jeleket tudás-, prompt- és dekódolási bizonytalanságra bontják.
12. A jeleket egy átlátható, tanult modellel vagy determinisztikus súlyozási szabállyal egyesítik.
13. Egy felügyelt kalibrátor opcionálisan transzformálja az egyesített valószínűséget.
14. A döntési politika alkalmazza a magabiztossági küszöböket, a feladatkritikusságot, az alcsoportkockázatot és az ellentmondási szabályokat.
15. A nyers és származtatott artefaktumokat szerializálják, mielőtt a tömör eredményt perzisztálják.
16. Az értékelési metrikákat, alcsoport-auditokat, szelektív görbéket, jelentéseket és driftmegfigyeléseket a perzisztált eredményekből számítják.

Megváltoztathatatlanság

A külső és kereszt-rétegű adatokat lefagyasztott Pydantic modellek és tiltott extra mezők reprezentálják. A módosítás a backend kliensekre, a kalibrátor tanító objektumokra, a relációs perzisztenciarekordokra és a behatárolt monitoringablakokra korlátozódik. A szerializált artefaktumokat kanonizálják és tartalom szerint címezik.

Backend izoláció

Minden backend ugyanazt a szinkron szerződést valósítja meg. A futtató újrapróbálkozásokat, az adapternek delegált timeoutokat, költségelszámolást, metrikákat és reprodukálhatósági rögzítést alkalmaz. A backend-specifikus payloadok nyers válaszokban és a reprodukálhatósági metaadatokban maradnak.

Hibahatárok

Egy sikertelen modellhívás backendhibát vált ki korlátozott számú újrapróbálkozás után. Egy sikertelen példa strukturált hibatípussal és üzenettel perzisztálódik. A befejezett eredményrekordokat későbbi hibakezelés soha nem írja felül. Az experiment állapota failed, amikor bármelyik példa sikertelen, miközben a sikeres példa artefaktumok továbbra is elérhetők ellenőrzésre és folytatásra.

Kalibrációs izoláció

A kalibrátor illesztése egy fejlesztési experimentet és egy félretett teszt experimentet igényel. A fúziós illesztés tréning-, validációs- és tesztexperimenteket igényel. A modellválasztás kizárólag validációt használ. A végső tesztértékelés a kiválasztás és az illesztés után történik, anélkül, hogy tesztcímkék bekerülnének az optimalizációba.

Tárolási modell

A relációs adatbázis tárolja az adathalmaz-manifesztumokat, a backendkonfigurációkat, az experiment állapotát, az eredményindexeket, a kalibrációs artefaktumokat és a drift-pillanatképeket. A nagy nyers rekordokat determinisztikus, tömörített JSON fájlokként tárolják. Az atomi csere megakadályozza a részben megírt artefaktumokat.

Biztonsági modell

A szolgáltatás API-kulcsos hitelesítést igényel, és kulcsonként korlátozza a kérések számát. A batch útvonalak egy konfigurált adatgyökérre korlátozottak. A naplók strukturáltak és maszkoltak. A nyers artefaktumok szándékosan teljesek, és telepítési szintű hozzáférés-vezérlést, titkosítást, megőrzési és biztonsági mentési szabályzatot igényelnek.
