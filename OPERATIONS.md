Műveletek

Adatbázis-migráció

alembic upgrade head

Az alap sémára csak akkor álljon vissza, amikor minden tárolt adat eltávolítható:

alembic downgrade base

Elérhetőség

A /health/live ellenőrzi a folyamat elérhetőségét. A /health/ready ellenőrzi az adatbázis-kapcsolatot, és jelzi a Redis-kapcsolatot. Az adatbázishiba a készenlétet nem elérhetőként jelöli. A Redis-hiba külön kerül jelentésre, mert a szolgáltatás megtart egy folyamaton belüli rate limitert.

Biztonsági mentés

A PostgreSQL-t és az artifact gyökeret egy kölcsönösen konzisztens ponton mentse le. Az adatbázis eredményrekordjai artifact útvonalakat tartalmaznak, ezért a teljes helyreállításhoz mindkét tároló szükséges. Az SQLite-telepítések esetén a beíró folyamatokat le kell állítani, vagy az SQLite online mentést kell használni az adatbázisfájl másolása előtt.

Visszaállítás ellenőrzése

Visszaállítás után:

alembic upgrade head
epistemic-uq backend list
curl http://localhost:8000/health/ready

Válasszon ki egy kísérletet, és ellenőrizze, hogy minden befejezett eredmény artifact útvonala létezik-e és olvasható-e. Generálja újra a jelentését, és hasonlítsa össze a jelentés JSON hashét a mentés előtti hashekkel, ha elérhetőek.

Skálázás

Több egyidejű worker esetén PostgreSQL-t használjon SQLite helyett. A Redis-t helyezze hitelesítés és hálózati szabályozás mögé. Futtasson több API-replikát terheléselosztó mögött. Az artifact-tárolónak megosztottnak, tartósnak kell lennie, és támogatnia kell az atomi átnevezés szemantikáját. Objektumtárolás esetén csatlakoztassa konzisztenciarétegen keresztül, vagy valósítson meg egy artifact-store adaptert feltételes írásokkal és tartalomhash-ekkel.

Kapacitás

A tárhely a modellhívások számával nő, nem csak a példák számával. Példánként és backendenként a hívások száma: egy alap hívás, egy opcionális önjelentés, egy opcionális igazságellenőrzés, a konfigurált mintaszám, valamint egy hívás perturbációnként. A modellek közötti végrehajtás ezt megszorozza a backendek számával. A token-nyomkövetések és a nyers szolgáltatói válaszok dominálhatják az artifact méretét.

Incidenskezelés

Backend hibaesetek megugrásakor vizsgálja meg az euq_model_calls_total metrikát backend és státusz szerint, majd a strukturált naplókat nyomkövetési azonosító alapján. Késleltetési változások esetén vizsgálja meg az euq_model_call_latency_seconds metrikát. Kalibrációs riasztások esetén ellenőrizze a tárolt drift pillanatképeket, és generálja újra az alcsoport-jelentéseket. Feldolgozási hibák esetén vizsgálja meg az euq_parsing_failures_total metrikát és a megfelelő auditgenerálási szöveget.

Kulcsrotáció

Állítsa be az EUQ_API_KEYS értékét egy vesszővel elválasztott halmazra, amely tartalmazza a régi és az új kulcsokat is, telepítse, migrálja az ügyfeleket, távolítsa el a régi kulcsot, majd telepítse újra. Soha ne írjon éles kulcsokat repository-fájlokba.
