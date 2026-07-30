Műveletek

Adatbázis-migráció

alembic upgrade head

Az alapértelmezett séma visszaminősítését csak akkor végezze el, ha minden tárolt adat eltávolítható:

alembic downgrade base

Elérhetőség

A /health/live a folyamat elérhetőségét ellenőrzi. A /health/ready az adatbázis-kapcsolatot ellenőrzi, és jelzi a Redis-kapcsolat állapotát. Az adatbázishiba a readiness állapotot nem elérhetőként jelöli. A Redis-hibát külön jelzi, mert a szolgáltatás megőrzi a folyamaton belüli rate limitert.

Mentés

A PostgreSQL-t és az artifact gyökeret egy kölcsönösen konzisztens ponton mentse le. Az adatbázis eredményrekordjai tartalmazzák az artifact elérési útjait, ამიტომ mindkét tárolóra szükség van a teljes helyreállításhoz. SQLite telepítések esetén a fájl másolása előtt le kell állítani az írókat, vagy használni kell a SQLite online biztonsági mentést.

Visszaállítás-ellenőrzés

Visszaállítás után:

alembic upgrade head
epistemic-uq backend list
curl http://localhost:8000/health/ready

Válasszon ki egy kísérletet, és ellenőrizze, hogy minden befejezett eredmény artifact elérési útja létezik és olvasható. Generálja újra a jelentését, és hasonlítsa össze a jelentés JSON hash-ét az előző mentés előtti hash-sel, ha elérhető.

Skálázás

Egyidejű munkafolyamatokhoz PostgreSQL-t használjon SQLite helyett. A Redis-t hitelesítés és hálózati szabályzat mögé helyezze. Futtasson több API-replikát terheléselosztó mögött. Az artifact-tárolónak megosztottnak, tartósnak kell lennie, és támogatnia kell az atomikus átnevezés szemantikáját. Objektumtárolás esetén csatlakoztassa konzisztenciarétegen keresztül, vagy valósítson meg egy artifact-store adaptert feltételes írásokkal és tartalomhash-ekkel.

Kapacitás

A tárhely a modellhívások számával növekszik, nem csupán a példák számával. Példánként és backendenként a hívások száma egy alap hívás, egy opcionális önbevallás, egy opcionális igazságellenőrzés, a konfigurált mintaszám, valamint egy hívás minden perturbációra. A modellek közötti végrehajtás ezt megszorozza a backendszámával. A tokennyomkövetések és a nyers szolgáltatói válaszok dominálhatják az artifact méretét.

Incidenskezelés

Backend hibacsúcsok esetén ellenőrizze az euq_model_calls_total metrikát backend és státusz szerint, majd vizsgálja meg a strukturált naplókat nyomkövetési azonosító alapján. Késleltetésváltozások esetén ellenőrizze az euq_model_call_latency_seconds metrikát. Kalibrációs riasztások esetén vizsgálja meg a tárolt drift pillanatképeket, és generálja újra az alcsoport-jelentéseket. Elemzési hibák esetén ellenőrizze az euq_parsing_failures_total metrikát és a megfelelő auditgenerálási szöveget.

Kulcsváltás

Állítsa be az EUQ_API_KEYS változót vesszővel elválasztott halmazra, amely tartalmazza a régi és az új kulcsot is, telepítse, migrálja az ügyfeleket, távolítsa el a régi kulcsot, majd telepítse újra. Soha ne írjon éles kulcsokat repository-fájlokba.
