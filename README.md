# mots-eveil

Un entraîneur de **mots d'éveil** pour [microWakeWord](https://github.com/kahrendt/microWakeWord)
(le moteur des Voice PE d'ESPHome et de Home Assistant), qui n'utilise **que des
voix de synthèse libres et des données libres**. Tu donnes une phrase
(« Hé Cumulus »), il rend un modèle `.tflite`, son manifeste `.json`, le bout de
YAML ESPHome et un rapport qui dit, chiffres à l'appui, **ce qui est mesuré et
surtout ce qui ne l'est pas**.

> **État : dépôt privé, pas encore audité pour publication.** Rien ici n'est
> public tant que le propriétaire du dépôt n'a pas donné son accord explicite. Licence du code :
> **à choisir** (voir plus bas).

## Ce que ça fait

1. **Préchargement** (une fois, dans `EVEIL_CACHE`) : échos de pièce, parole
   française lue, bruits et musique libres.
2. **Synthèse** de la phrase (plusieurs graphies, vitesses et bruits) avec les
   voix libres de [Piper](https://github.com/OHF-Voice/piper1-gpl), plus des
   phrases « qui ne sont pas le mot » dites par les mêmes voix, pour que le
   modèle n'apprenne pas « la voix de synthèse » au lieu du mot.
3. **Spectrogrammes** microWakeWord avec bruit et écho.
4. **Entraînement** sur une carte NVIDIA (2 variantes, la meilleure sur la
   *sélection* est livrée, jamais jugée sur le test).
5. **Évaluation** par `pymicro-wakeword` sur le son brut, comme l'appareil le
   fait : le seuil est un *cran* (`int(p × 255)`) comparé à la somme des N
   dernières sorties. Fausses détections par heure sur de la parole française
   et des bruits jamais entendus à l'entraînement ; le seuil et les trois
   sensibilités du Voice PE sont choisis par la cible (0,5 fausse détection par
   heure par défaut, prouvée par `(n + 3) / heures`).
6. **Livraison** dans `EVEIL_DONNEES/modeles/<mot>/<version>/` : `<mot>.tflite`,
   `<mot>.json`, `voice-pe.yaml`, `rapport.md`, `rapport.json`.

## Utilisation

```bash
./installer.sh                      # venv + microWakeWord au commit épinglé
mkdir -p donnees/mots/cumulus
cat > donnees/mots/cumulus/fiche.json <<'JSON'
{"phrase": "Hé Cumulus",
 "variantes": ["Hey Cumulus", "Eh Cumulus"],
 "negatifs_proches": ["Cumulus", "Hé Cumul", "Hé Romulus", "Hé plus", "Un cumulus"],
 "cible_fausses_detections_h": 0.5}
JSON
EVEIL_DONNEES=./donnees EVEIL_CACHE=./cache EVEIL_MOT=cumulus \
  venv-eveil/bin/python -u entraineur/entrainer.py          # --essai pour une petite chaîne de test
```

Il faut TensorFlow 2.21 (l'image `tensorflow/tensorflow:2.21.0-gpu` convient),
une carte NVIDIA avec ~4 Go libres (`EVEIL_GPU` force la carte), ~100 Go de
cache et plusieurs heures. `--sans-gpu` existe, mais est très lent.

Variables : `EVEIL_DONNEES`, `EVEIL_CACHE`, `EVEIL_MOT`, `EVEIL_GPU`,
`EVEIL_TF_MEMOIRE_MIO`, `EVEIL_GARDER_TRAVAIL=1` (débogage).

## La garde

L'entraîneur refuse de démarrer si son dossier de données contient `prises/`,
`ambiance/`, `reveils/` ou `repas/` : ces noms sont ceux d'un entraîneur
*privé* (vraies voix, ambiance d'une maison) dont celui-ci est la version
publique. Trois protections indépendantes : la garde (`garde_public`, appelée
au début de `main` et d'`entrainer_mot`), l'absence de tout chemin qui lise ces
dossiers, et `entraineur/test_public.py` (CI `garde`).

## Sources de données et voix libres

| Source | Rôle | Licence |
|---|---|---|
| [MUSAN](https://www.openslr.org/17/) (`noise`, `music`) | bruits et musique de fond | CC BY 4.0 |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) voix `fr_FR-mls-medium` (125 locuteurs) | voix de synthèse | CC BY 4.0 (données MLS) ; Piper lui-même : GPL-3.0 (outil lancé dans un venv à part, pas le modèle) |
| Piper voix `fr_FR-siwis-medium` (1 locuteur) | voix de synthèse | CC BY 4.0 (données SIWIS) |
| [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) français | parole française lue (apprentissage et mesure) | CC BY 4.0 |
| [Mozilla Common Voice 17](https://commonvoice.mozilla.org/) français (miroir Hugging Face) | parole de milliers de locuteurs, dont québécois | CC0 1.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) `fr_fr` | parole française lue | CC BY 4.0 |
| [DEMAND](https://zenodo.org/records/1227121) | bruits de pièces et d'extérieur | CC BY 4.0 |
| [MIT IR Survey](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) | échos de pièce | **À CONFIRMER** : la fiche Hugging Face dit « unknown » et la page du MIT n'en dit rien ; CC BY 4.0 annoncé ailleurs, non vérifié |

À vérifier avant toute publication (voir « Avant de rendre public ») : la licence
des échos MIT, et le fait que la partie `music` de MUSAN ne mélange pas des
licences incompatibles.

## Données écartées, et pourquoi

| Donnée | Pourquoi elle n'y est pas |
|---|---|
| Négatifs « officiels » de microWakeWord | Fiche Hugging Face `cc-by-nc-4.0` (non commercial), bâtis sur CHiME6, FMA, FSD50K et WHAM : un modèle distribué n'en dérive pas |
| AudioSet | Audio YouTube : pas redistribuable ; seules les étiquettes sont CC BY |
| FMA (Free Music Archive, `fma_xsmall`) | Licences variées, dont NC |
| Vraies voix, ambiance de maison, réveils d'appareil | Privées : elles n'entrent jamais ici (la garde) |
| Voix Google Cloud TTS fr-CA | Disponibles en **option** seulement (`EVEIL_GOOGLE_CLE` **et** `EVEIL_PUBLIC_GOOGLE=1`) : les conditions de Google ne sont pas tranchées pour un modèle distribué ; le rapport le dit quand elles servent |

## Ce que les essais ont donné

Trois essais sur le même mot « Hé Cumulus », une carte graphique, avec les
rapports tels que produits dans [`rapports/`](rapports/) :

| | Essai 1 | Essai 2 (libre améliorée) | Essai 3 (essai 2 + Google fr-CA) |
|---|---|---|---|
| Fausses détections / h (son long libre) | 2,00 | **0,39** | 0,68 |
| Voix de synthèse gardées de côté, entendues | 65 % | 88 % | 72 % |
| Phrases proches qui déclenchent | 17/133 | 3/133 | 7/192 |
| Heures de son long mesurées | 22,5 | 31,0 | 31,0 |
| Cible 0,5 / h | non | **oui** | non |

Lire avec la réserve que les rapports eux-mêmes mettent en tête : **aucune vraie
voix du mot** n'est dans l'entraînement ni dans la mesure (le taux « entendu » est
optimiste), **aucune ambiance de maison** ni micro réel (les vraies fausses
détections seront probablement plus nombreuses), et les voix sont celles de la
France (Piper), pas du français québécois. Un modèle d'éveil entraîné ainsi
rate des voix réelles que celui entraîné avec de vraies prises attrape :
teste-le chez toi et garde l'ancien réglage sous la main. La cible de l'essai 2
est tenue sur de la parole lue et des bruits, pas sur ta maison.

Le premier rapport dit « pas assez d'heures pour prouver la cible » alors que la
cible n'était pas tenue (2,00 > 0,5) : corrigé depuis dans `rapport_public_md`.

## Licence

Le code n'a **pas encore de licence** : choix en attente (MIT proposé par défaut).
Sans fichier `LICENSE`, personne n'a le droit de le réutiliser. Les données et
les voix gardent leurs propres licences (tableau ci-dessus), et un modèle produit
en hérite : le rapport de chaque entraînement les liste.

## Avant de rendre public

- [ ] accord explicite du propriétaire ;
- [ ] licence du code choisie (`LICENSE`) ;
- [ ] licence des échos MIT confirmée ; MUSAN `music` vérifié ;
- [ ] audit refait sur le dépôt entier et son historique.
