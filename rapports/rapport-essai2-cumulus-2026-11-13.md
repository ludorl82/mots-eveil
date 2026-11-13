# Hé Cumulus — modèle de mot d'éveil, entraînement 20261112-2345

**Lis ceci d'abord.** Ce modèle a été entraîné **uniquement sur des voix de synthèse** et des données libres, sans une seule vraie voix humaine disant le mot. Les chiffres ci-dessous ne disent donc PAS comment il se comporte avec ta voix, ta pièce et ton micro : teste-le chez toi, et garde l'ancien réglage sous la main.

## Ce qui est mesuré

Cible : **0.5 fausse détection par heure** au plus, mesurée sur du son long LIBRE : parole française lue (MLS) et bruits / musique (MUSAN), jamais entendus à l'entraînement.

| Mesure | Résultat |
|---|---|
| Fenêtre glissante / cran choisi | 10 sorties / 239 (seuil 0.937) |
| Voix de synthèse gardées de côté, entendues | 88 % |
| Fausses détections par heure (son long libre) | 0.39 |
| Heures de son long mesurées | 31.00 h (fond_libre 2.86 h, parole_fr 20.14 h, parole_cv 8.00 h) |
| Phrases proches qui déclenchent | 3/133 |

## Ce qui n'est PAS mesuré

- Aucune vraie voix du mot : le taux « entendu » vient de la même famille de voix de synthèse que l'entraînement (voix différentes d'un côté et de l'autre du partage, mais toutes synthétiques). Il est optimiste.
- Aucune ambiance de maison (télé, repas, enfants) ni ton micro : les fausses détections réelles seront probablement plus nombreuses que ci-dessus.
- Voix françaises de France (Piper mls / siwis), pas du français québécois : l'accent n'est pas couvert sauf si l'option Google fr-CA a été activée (voir les sources).

## Variantes essayées

- base (livrée) : fenêtre 10, seuil 0.918, 0.60 h-GPU
- prudent : fenêtre 7, seuil 0.976, 0.59 h-GPU

## Ce qui a servi

- Synthèse : moteur piper, 126 voix, 5760 prises du mot, 630 phrases proches.
- Aucune vraie prise, aucune ambiance de la maison, aucun réveil d'appareil.
- Spectrogrammes : {"positifs_synthese": {"training": 80380, "validation": 6110, "testing": 1130}, "proches": {"training": 8820, "validation": 560, "testing": 133}, "negatifs_synthese": {"training": 203280, "validation": 15030, "testing": 2949}, "fond_libre": {"training": 985, "validation": 2417}, "parole_fr": {"training": 41756}}
- Heures-GPU : 1.19

## Sources et licences

- rir : échos de pièce — CC BY 4.0 selon la documentation de l'entraîneur ; la fiche Hugging Face dit « unknown » et la page du MIT n'en dit rien : À CONFIRMER
- mls_fr : parole française lue — CC BY 4.0
- musan : bruits et musique de fond — CC BY 4.0 (noise, music)
- demand : bruits de pièces et d'extérieur, 16 kHz — CC BY 4.0 (DEMAND, Zenodo 1227121)
- common_voice_fr : parole française de milliers de locuteurs, dont québécois — CC0 1.0 (Mozilla Common Voice 17, miroir Hugging Face)
- fleurs_fr : parole française lue — CC BY 4.0 (FLEURS, fr_fr)
- piper_mls : voix de synthèse libres (125 locuteurs) — CC BY 4.0 (MLS) ; Piper : GPL-3.0 (outil, pas le modèle)
- piper_siwis : voix de synthèse libre (1 locuteur) — CC BY 4.0 (SIWIS) ; Piper : GPL-3.0 (outil, pas le modèle)

Écarté exprès de ce mode : AudioSet (audio YouTube), FMA (licences NC) et les négatifs officiels de microWakeWord (cc-by-nc-4.0).

## Comparaison (même mot, même rapport)

| | Essai 1 | Essai 2 (libre améliorée) | Essai 3 (essai 2 + Google fr-CA) |
|---|---|---|---|
| Fausses détections / h | 2.00 | **0.39** | 0.68 |
| Voix de synthèse gardées entendues | 65 % | 88 % | 72 % |
| Phrases proches qui déclenchent | 17/133 | 3/133 | 7/192 |
| Heures de son long mesurées | 22.5 | 31.0 | 31.0 |
| Cible 0.5 / h | non | **oui** | non |

Réserve : la cible est tenue sur du son libre (parole lue + bruits), pas sur une vraie maison.
