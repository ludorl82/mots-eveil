> Essai à la main du 2026-11-12 (une carte graphique, 1 variante, 0,30 h-GPU), rapport tel que produit par
> la version du commit précédent. Son avertissement « pas assez d'heures » est en réalité
> « cible NON tenue » (2,00 > 0,5) : corrigé depuis dans `rapport_public_md`.

# Hé Cumulus — modèle de mot d'éveil, entraînement 20261112-2135

**Lis ceci d'abord.** Ce modèle a été entraîné **uniquement sur des voix de synthèse** et des données libres, sans une seule vraie voix humaine disant le mot. Les chiffres ci-dessous ne disent donc PAS comment il se comporte avec ta voix, ta pièce et ton micro : teste-le chez toi, et garde l'ancien réglage sous la main.

## Ce qui est mesuré

Cible : **0.5 fausse détection par heure** au plus, mesurée sur du son long LIBRE : parole française lue (MLS) et bruits / musique (MUSAN), jamais entendus à l'entraînement.

| Mesure | Résultat |
|---|---|
| Fenêtre glissante / cran choisi | 5 sorties / 254 (seuil 0.996) |
| Voix de synthèse gardées de côté, entendues | 65 % |
| Fausses détections par heure (son long libre) | 2.00 |
| Heures de son long mesurées | 22.50 h (fond_libre 2.36 h, parole_fr 20.14 h) |
| Phrases proches qui déclenchent | 17/133 |

**Attention : pas assez d'heures pour prouver la cible** (il en faut au moins 6 sans fausse détection). Le seuil est mis au plus prudent.

## Ce qui n'est PAS mesuré

- Aucune vraie voix du mot : le taux « entendu » vient de la même famille de voix de synthèse que l'entraînement (voix différentes d'un côté et de l'autre du partage, mais toutes synthétiques). Il est optimiste.
- Aucune ambiance de maison (télé, repas, enfants) ni ton micro : les fausses détections réelles seront probablement plus nombreuses que ci-dessus.
- Voix françaises de France (Piper mls / siwis), pas du français québécois : l'accent n'est pas couvert sauf si l'option Google fr-CA a été activée (voir les sources).

## Variantes essayées

- base (livrée) : fenêtre 5, seuil 0.996, 0.30 h-GPU

## Ce qui a servi

- Synthèse : moteur piper, 126 voix, 5760 prises du mot, 630 phrases proches.
- Aucune vraie prise, aucune ambiance de la maison, aucun réveil d'appareil.
- Spectrogrammes : {"positifs_synthese": {"training": 80380, "validation": 6110, "testing": 1130}, "proches": {"training": 8820, "validation": 560, "testing": 133}, "fond_libre": {"training": 949, "validation": 1526}, "parole_fr": {"training": 9762}}
- Heures-GPU : 0.30

## Sources et licences

- rir : échos de pièce — CC BY 4.0 selon la documentation de l'entraîneur ; la fiche Hugging Face dit « unknown » et la page du MIT n'en dit rien : À CONFIRMER
- mls_fr : parole française lue — CC BY 4.0
- musan : bruits et musique de fond — CC BY 4.0 (noise, music)
- piper_mls : voix de synthèse libres (125 locuteurs) — CC BY 4.0 (MLS) ; Piper : GPL-3.0 (outil, pas le modèle)
- piper_siwis : voix de synthèse libre (1 locuteur) — CC BY 4.0 (SIWIS) ; Piper : GPL-3.0 (outil, pas le modèle)

Écarté exprès de ce mode : AudioSet (audio YouTube), FMA (licences NC) et les négatifs officiels de microWakeWord (cc-by-nc-4.0).
