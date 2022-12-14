from typing import Dict


SMILER_VERBALIZERS: Dict[str, Dict[str, str]] = {
    "en": {
        "birth-place": "birth place",
        "eats": "eats",
        "event-year": "event year",
        "first-product": "first product",
        "from-country": "from country",
        "has-author": "has author",
        "has-child": "has child",
        "has-edu": "has education",
        "has-genre": "has genre",
        "has-height": "has height",
        "has-highest-mountain": "has highest mountain",
        "has-length": "has length",
        "has-lifespan": "has lifespan",
        "has-nationality": "has nationality",
        "has-occupation": "has occupation",
        "has-parent": "has parent",
        "has-population": "has population",
        "has-sibling": "has sibling",
        "has-spouse": "has spouse",
        "has-tourist-attraction": "has tourist attraction",
        "has-type": "has type",
        "has-weight": "has weight",
        "headquarters": "headquarters",
        "invented-by": "invented by",
        "invented-when": "invented when",
        "is-member-of": "is member of",
        "is-where": "located in",
        "loc-leader": "location leader",
        "movie-has-director": "movie has director",
        "no_relation": "no relation",
        "org-has-founder": "organization has founder",
        "org-has-member": "organization has member",
        "org-leader": "organization leader",
        "post-code": "post code",
        "starring": "starring",
        "won-award": "won award",
    },
    "de": {
        "birth-place": "Geburtsort",
        "event-year": "Veranstaltungsjahr",
        "from-country": "vom Land",
        "has-author": "hat Autor",
        "has-child": "hat Kind",
        "has-edu": "hat Bildung",
        "has-genre": "hat Genre",
        "has-occupation": "hat Beruf",
        "has-parent": "hat Elternteil",
        "has-population": "hat Bevölkerung",
        "has-spouse": "hat Ehepartner",
        "has-type": "hat Typ",
        "headquarters": "Hauptsitz",
        "is-member-of": "ist Mitglied von",
        "is-where": "gelegen in",
        "loc-leader": "Standortleiter",
        "movie-has-director": "Film hat Regisseur",
        "no_relation": "keine Beziehung",
        "org-has-founder": "Organisation hat Gründer",
        "org-has-member": "Organisation hat Mitglied",
        "org-leader": "Organisationsleiter",
        "won-award": "gewann eine Auszeichnung",
    },
    "es": {
        "birth-place": "lugar de nacimiento",
        "event-year": "año del evento",
        "from-country": "del país",
        "has-author": "tiene autor",
        "has-child": "tiene hijo",
        "has-edu": "tiene educación",
        "has-genre": "tiene género",
        "has-occupation": "tiene ocupación",
        "has-parent": "tiene padre",
        "has-population": "tiene población",
        "has-spouse": "tiene cónyuge",
        "has-type": "tiene tipo",
        "headquarters": "sede central",
        "is-member-of": "es miembro de",
        "is-where": "situado en",
        "loc-leader": "líder de ubicación",
        "movie-has-director": "película cuenta con el director",
        "no_relation": "sin relación",
        "org-has-founder": "organización cuenta con el fundador",
        "org-has-member": "organización tiene miembro",
        "won-award": "ganó el premio",
    },
    "ar": {
        "event-year": "سنة الحدث",
        "has-edu": "لديه تعليم",
        "has-genre": "له النوع",
        "has-occupation": "لديه احتلال",
        "has-population": "عدد السكان",
        "has-type": "لديه نوع",
        "is-member-of": "عضو في",
        "no_relation": "لا علاقة",
        "won-award": "فاز بالجائزة",
    },
    "fa": {
        "event-year": "سال رویداد",
        "has-edu": "تحصیلات دارد",
        "has-genre": "ژانر دارد",
        "has-occupation": "شغل دارد",
        "has-population": "جمعیت دارد",
        "has-type": "نوع دارد",
        "is-member-of": "عضو است",
        "no_relation": "هیچ رابطه ای",
    },
    "fr": {
        "birth-place": "lieu de naissance",
        "event-year": "année de l'événement",
        "from-country": "du pays",
        "has-author": "a un auteur",
        "has-child": "a un enfant",
        "has-edu": "a une éducation",
        "has-genre": "a un genre",
        "has-occupation": "a une profession",
        "has-parent": "a un parent",
        "has-population": "a de la population",
        "has-spouse": "a un conjoint",
        "has-type": "a le type",
        "headquarters": "siège social",
        "is-member-of": "est membre de",
        "is-where": "situé à",
        "loc-leader": "guide d'emplacement",
        "movie-has-director": "le film a un réalisateur",
        "no_relation": "aucune relation",
        "org-has-founder": "l'organisation a un fondateur",
        "org-has-member": "l'organisation a un membre",
        "org-leader": "chef d'organisation",
        "won-award": "a remporté le prix",
    },
    "it": {
        "birth-place": "luogo di nascita",
        "event-year": "anno dell'evento",
        "from-country": "dal paese",
        "has-author": "ha autore",
        "has-child": "ha un figlio",
        "has-edu": "ha un'educazione",
        "has-genre": "ha genere",
        "has-occupation": "ha occupazione",
        "has-parent": "ha un genitore",
        "has-population": "ha una popolazione",
        "has-spouse": "ha un coniuge",
        "has-type": "ha il tipo",
        "headquarters": "sede centrale",
        "is-member-of": "è membro di",
        "is-where": "situato in",
        "loc-leader": "leader della posizione",
        "movie-has-director": "il film ha direttore",
        "no_relation": "nessuna relazione",
        "org-has-founder": "l'organizzazione ha fondatore",
        "org-has-member": "l'organizzazione ha un membro",
        "org-leader": "leader dell'organizzazione",
        "won-award": "ha vinto un premio",
    },
    "ko": {
        "birth-place": "출생지",
        "event-year": "이벤트 연도",
        "first-product": "첫 번째 제품",
        "from-country": "나라에서",
        "has-author": "저자가 있다",
        "has-child": "아이가 있다",
        "has-edu": "교육이 있다",
        "has-genre": "장르가 있다",
        "has-highest-mountain": "가장 높은 산이 있다",
        "has-nationality": "국적이 있다",
        "has-occupation": "직업이 있다",
        "has-parent": "부모가 있다",
        "has-population": "인구가 있다",
        "has-sibling": "형제가 있다",
        "has-spouse": "배우자가 있다",
        "has-tourist-attraction": "관광명소가 있다",
        "has-type": "유형이 있습니다",
        "headquarters": "본부",
        "invented-by": "에 의해 발명",
        "invented-when": "언제 발명",
        "is-member-of": "의 회원입니다",
        "is-where": "어디에",
        "movie-has-director": "영화에 감독이 있다",
        "no_relation": "관계가 없다",
        "org-has-founder": "조직에는 설립자가 있습니다",
        "org-has-member": "조직에 구성원이 있습니다",
        "org-leader": "조직 리더",
        "won-award": "수상",
    },
    "nl": {
        "birth-place": "geboorteplaats",
        "event-year": "evenementenjaar",
        "from-country": "van het land",
        "has-author": "heeft auteur",
        "has-child": "heeft kind",
        "has-edu": "heeft onderwijs",
        "has-genre": "heeft genre",
        "has-occupation": "heeft beroep",
        "has-parent": "heeft ouder",
        "has-population": "heeft bevolking",
        "has-spouse": "heeft echtgenoot",
        "has-type": "heeft type",
        "headquarters": "hoofdkantoor",
        "is-member-of": "is lid van",
        "is-where": "gevestigd in",
        "loc-leader": "locatieleider",
        "movie-has-director": "film had regisseur",
        "no_relation": "geen relatie",
        "org-has-founder": "organisatie heeft oprichter",
        "org-has-member": "organisatie heeft lid",
        "org-leader": "organisatieleider",
        "won-award": "won prijs",
    },
    "pl": {
        "birth-place": "miejsce urodzenia",
        "event-year": "rok imprezy",
        "from-country": "z kraju",
        "has-author": "ma autor",
        "has-child": "ma dziecko",
        "has-edu": "ma wykształcenie",
        "has-genre": "ma gatunek",
        "has-occupation": "ma zawód",
        "has-parent": "ma rodzica",
        "has-population": "ma ludność",
        "has-spouse": "ma współmałżonka",
        "has-type": "ma typ",
        "headquarters": "siedziba główna",
        "is-member-of": "jest członkiem",
        "is-where": "mieszczący się w",
        "loc-leader": "lider lokalizacji",
        "movie-has-director": "film ma reżysera",
        "org-has-founder": "organizacja ma założyciela",
        "org-has-member": "organizacja ma członków",
        "org-leader": "lider organizacji",
        "won-award": "otrzymał nagrodę",
    },
    "pt": {
        "birth-place": "local de nascimento",
        "event-year": "ano do evento",
        "from-country": "do país",
        "has-author": "tem autor",
        "has-child": "tem filho",
        "has-edu": "tem educação",
        "has-genre": "tem género",
        "has-occupation": "tem ocupação",
        "has-parent": "tem pai",
        "has-population": "tem população",
        "has-spouse": "tem cônjuge",
        "has-type": "tem tipo",
        "headquarters": "sede",
        "is-member-of": "é membro de",
        "is-where": "localizado em",
        "loc-leader": "loc leader",
        "movie-has-director": "filme tem realizador",
        "no_relation": "sem relação",
        "org-has-founder": "organização tem fundador",
        "org-has-member": "organização tem membro",
        "org-leader": "líder da organização",
        "won-award": "ganhou prémio",
    },
    "ru": {
        "event-year": "год события",
        "has-edu": "имеет образование",
        "has-genre": "имеет жанр",
        "has-occupation": "имеет профессию",
        "has-population": "имеет население",
        "has-type": "имеет тип",
        "is-member-of": "является членом",
        "no_relation": "без связи",
    },
    "sv": {
        "birth-place": "födelseort",
        "event-year": "År för evenemanget",
        "from-country": "från ett land",
        "has-author": "har en författare",
        "has-child": "har chili",
        "has-edu": "har utbildning",
        "has-genre": "har en genre",
        "has-occupation": "har ockuperat",
        "has-parent": "har en förälder",
        "has-population": "har en befolkning",
        "has-spouse": "har make eller maka",
        "has-type": "har typ",
        "headquarters": "huvudkontor",
        "is-member-of": "är medlem i",
        "is-where": "som ligger i",
        "loc-leader": "platsansvarig",
        "movie-has-director": "filmen har regissör",
        "no_relation": "ingen relation",
        "org-has-founder": "organisationen har en grundare",
        "org-has-member": "organisationen har en medlem",
        "org-leader": "ledare för organisationen",
        "won-award": "vann ett pris",
    },
    "uk": {
        "event-year": "рік події",
        "has-edu": "має освіту",
        "has-genre": "має жанр",
        "has-occupation": "має заняття",
        "has-population": "має населення",
        "has-type": "має тип",
        "no_relation": "ніякого відношення",
    },
}
