"""Entraîneur de mots d'éveil libres : n'importe quel mot, à partir de voix de
synthèse libres (Piper) et de données libres seulement.
Sortie : un modèle microWakeWord (le moteur des Voice PE d'ESPHome), son
manifeste, le bout de YAML ESPHome et un rapport qui dit, chiffres à l'appui,
ce qui est mesuré et surtout ce qui ne l'est PAS.

Le déroulement, pour chaque mot (mots/<mot>/fiche.json, ou EVEIL_MOT) :

  1. préchargement, une fois pour toutes, dans le cache (EVEIL_CACHE) :
     échos de pièce (MIT), parole française lue (MLS, Common Voice, FLEURS),
     bruits et musique libres (MUSAN, DEMAND) ;
  2. synthèse libre (Piper : voix mls et siwis), plusieurs graphies, vitesses
     et bruits ; mise en cache. Les voix Google fr-CA ne servent qu'en OPTION ;
  3. caractéristiques : spectrogrammes microWakeWord, avec bruit et écho ;
  4. entraînement sur UNE carte graphique (ou le processeur, très lent) ;
  5. évaluation par pymicro-wakeword sur le son brut, comme sur l'appareil :
     fausses détections par heure sur de la parole française et des bruits
     jamais entendus à l'entraînement ; le seuil et les trois sensibilités du
     Voice PE sont choisis par la cible de fausses détections par heure ;
  6. livraison dans EVEIL_DONNEES/modeles/<mot>/<version>/.

Aucune vraie voix, aucune ambiance de maison : la garde (garde_public) refuse
de démarrer si le dossier de données contient prises/, ambiance/, reveils/ ou
repas/. Voir le README.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("entraineur")

DONNEES = Path(os.environ.get("EVEIL_DONNEES", "/donnees"))
CACHE = Path(os.environ.get("EVEIL_CACHE", "/cache"))
GOOGLE_CLE = os.environ.get("EVEIL_GOOGLE_CLE", "").strip()
TAUX = 16000
FENETRE_ARRET_S = 1.0  # après une détection, l'appareil se tait (réfractaire)

HF = "https://huggingface.co/datasets"
SOURCES = {
    # nom : (url, licence, rôle)
    "rir": (f"{HF}/davidscripka/MIT_environmental_impulse_responses", "CC BY 4.0", "échos de pièce"),
    "mls_fr": (f"{HF}/facebook/multilingual_librispeech/resolve/main/french/", "CC BY 4.0", "parole française lue"),
}

# --- Ce qui est permis, et ce qui ne l'est pas ---------------------------------------
# Une recette qui ne sait pas où sont de vraies voix. Jamais une vraie prise, une
# ambiance de maison ou un réveil d'appareil dans un modèle destiné à d'autres :
# ni lue, ni mesurée, ni mélangée. Trois protections, indépendantes :
#   1. la garde : le dossier de données ne doit contenir AUCUN dossier privé ;
#   2. le code n'a aucun chemin qui lise prises/, ambiance/ ou reveils/ ;
#   3. test_public.py rejoue 1 et 2 à chaque PR.
# Écartés exprès (licences non redistribuables) : les négatifs officiels de
# microWakeWord (cc-by-nc-4.0, bâtis sur CHiME6, FMA, FSD50K et WHAM), AudioSet
# (audio YouTube) et FMA (licences NC) : un modèle distribué n'en dérive pas.
DOSSIERS_PRIVES = ("prises", "ambiance", "reveils", "repas")
SOURCES_REFUSEES_EN_PUBLIC = ("audioset", "fma", "negatifs_officiels")
HFM = "https://huggingface.co"
PIPER_VERSION = "1.8.0"
PIPER_VOIX = {
    # nom : (url du modèle, locuteurs, licence des données)
    "mls": (f"{HFM}/rhasspy/piper-voices/resolve/main/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx", 125, "CC BY 4.0 (MLS)"),
    "siwis": (f"{HFM}/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx", 1, "CC BY 4.0 (SIWIS)"),
}
SOURCES_LIBRES = {
    "rir": (SOURCES["rir"][0], "CC BY 4.0 selon une source secondaire non vérifiée ; la fiche Hugging Face dit « unknown » et la page du MIT n'en dit rien : À CONFIRMER", "échos de pièce"),
    "mls_fr": SOURCES["mls_fr"],
    "musan": ("https://www.openslr.org/resources/17/musan.tar.gz", "CC BY 4.0 (noise, music)", "bruits et musique de fond"),
    "demand": ("https://zenodo.org/records/1227121", "CC BY 4.0 (DEMAND, Zenodo 1227121)", "bruits de pièces et d'extérieur, 16 kHz"),
    "common_voice_fr": (f"{HFM}/datasets/fsicoli/common_voice_17_0", "CC0 1.0 (Mozilla Common Voice 17, miroir Hugging Face)", "parole française de milliers de locuteurs, dont québécois"),
    "fleurs_fr": (f"{HFM}/datasets/google/fleurs", "CC BY 4.0 (FLEURS, fr_fr)", "parole française lue"),
    "piper_mls": (PIPER_VOIX["mls"][0], PIPER_VOIX["mls"][2] + " ; Piper : GPL-3.0 (outil, pas le modèle)", "voix de synthèse libres (125 locuteurs)"),
    "piper_siwis": (PIPER_VOIX["siwis"][0], PIPER_VOIX["siwis"][2] + " ; Piper : GPL-3.0 (outil, pas le modèle)", "voix de synthèse libre (1 locuteur)"),
}


class ErreurPublique(RuntimeError):
    """Une donnée privée est à portée du mode public : on s'arrête net."""


def garde_public(donnees=None):
    """Refuse de démarrer si le dossier de données contient un dossier privé.
    Appelée avant tout autre geste (main) ET en tête d'entrainer_mot."""
    racine = Path(donnees if donnees is not None else DONNEES)
    trouves = [d for d in DOSSIERS_PRIVES if (racine / d).exists()]
    if trouves:
        raise ErreurPublique(f"refusé : {racine} contient {', '.join(trouves)} (données privées). "
                             "Cet entraîneur tourne avec un dossier de données à lui, sans aucun de ces dossiers.")
    return racine


# --- Petits outils ----------------------------------------------------------------


def empreinte(texte, n=12):
    return hashlib.sha256(texte.encode()).hexdigest()[:n]


def cote(chemin):
    """entrainement | validation | test, stable pour un même fichier."""
    h = int(hashlib.sha256(Path(chemin).name.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 20 else "validation" if h < 30 else "entrainement"


def lire_json(p, defaut=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaut


def ecrire_json(p, valeur):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(valeur, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def telecharger(url, cible):
    cible = Path(cible)
    if cible.exists():
        return cible
    cible.parent.mkdir(parents=True, exist_ok=True)
    part = cible.with_suffix(cible.suffix + ".part")
    log.info("téléchargement %s", url)
    for essai in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, cible)
            return cible
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log.warning("échec (%s), nouvel essai", e)
            time.sleep(10 * (essai + 1))
    raise RuntimeError(f"téléchargement impossible : {url}")


def lire_audio(source):
    """float32 mono à 16 kHz, depuis un chemin ou des octets (wav, flac, mp3…)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    donnees, taux = sf.read(io.BytesIO(source) if isinstance(source, bytes) else str(source),
                            dtype="float32", always_2d=True)
    son = donnees.mean(axis=1)
    if taux != TAUX:
        g = math.gcd(int(taux), TAUX)
        son = resample_poly(son, TAUX // g, int(taux) // g).astype(np.float32)
    return son


def ecrire_wav(chemin, son):
    from scipy.io import wavfile

    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(chemin), TAUX, (np.clip(son, -1, 1) * 32767).astype(np.int16))


def fait(dossier):
    return (Path(dossier) / ".fait").exists()


def marquer(dossier):
    Path(dossier).mkdir(parents=True, exist_ok=True)
    (Path(dossier) / ".fait").write_text(dt.datetime.now().isoformat())


# --- 1. Préchargement -------------------------------------------------------------




def _precharger_rir():
    rir = CACHE / "rir"
    if not fait(rir):
        api = "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz"
        with urllib.request.urlopen(api, timeout=60) as r:
            fichiers = [x["path"] for x in json.load(r) if x["path"].endswith(".wav")]
        for f in fichiers:
            telecharger(f"{SOURCES['rir'][0]}/resolve/main/{f}", rir / Path(f).name)
        marquer(rir)


def _precharger_parole(leger):
    # Parole française lue (MLS, locuteurs différents d'une partie à l'autre) :
    #   entrainement + entrainement_plus (~40 h) pour apprendre à ne pas réagir ;
    #   selection (6 h d'un autre morceau) pour choisir entre les variantes ;
    #   eval + test (dev et test, ~20 h) pour mesurer et fixer le seuil.
    # 8,5 h ne laissaient droit qu'à une fausse détection pour prouver 0,5/h
    # (2026-11-04) ; 20 h en laissent 7. En léger : « 1_hours » seulement.
    parties = [("entrainement", "1_hours" if leger else "9_hours", None), ("eval", "1_hours" if leger else "dev", None)]
    if not leger:
        parties += [("entrainement_plus", "train-00000-of-00034", None), ("selection", "train-00033-of-00034", 6.0),
                    ("test", "test-00000-of-00001", None)]
    for partie, fichier, heures_max in parties:
        d = CACHE / "parole_fr" / partie
        if not fait(d):
            import pyarrow.parquet as pq

            if "-of-" not in fichier:
                fichier += "-00000-of-00001"
            p = telecharger(SOURCES["mls_fr"][0] + f"{fichier}.parquet", CACHE / "brut" / f"mls_fr_{fichier}.parquet")
            secondes = 0.0
            for i, a in enumerate(x for lot in pq.ParquetFile(p).iter_batches(columns=["audio"], batch_size=64)
                                  for x in lot.column("audio").to_pylist()):
                if heures_max and secondes >= heures_max * 3600:
                    break
                if leger and partie == "eval" and i % 2 == 0:
                    continue
                if leger and partie == "entrainement" and i % 2 == 1:
                    continue
                try:
                    son = lire_audio(a["bytes"])
                    ecrire_wav(d / f"{i:06d}.wav", son)
                    secondes += len(son) / TAUX
                except Exception as e:
                    log.warning("mls %d ignoré : %s", i, e)
            marquer(d)
            if "train-" in fichier:
                p.unlink(missing_ok=True)  # 500 Mo de parquet, déjà convertis


def precharger_public(leger=False):
    """Le préchargement du mode public : échos (MIT), parole française lue
    (MLS), bruits et musique libres (MUSAN). Ni AudioSet, ni FMA, ni les
    négatifs officiels de microWakeWord (licences NC ou non redistribuables)."""
    _precharger_rir()
    _precharger_musan(leger)
    _precharger_demand(leger)
    _precharger_parole(leger)
    _precharger_parole_plus(leger)
    _precharger_cv(leger)
    _precharger_fleurs(leger)


# Des phrases qui ne sont PAS le mot : le modèle ne doit pas apprendre « la voix
# de synthèse » au lieu du mot. Dites par les MÊMES voix que les positifs
# (premier essai public, 2026-11-12 : positifs 100 % synthèse, négatifs 100 %
# voix humaines ou bruit, 2,0 fausses détections par heure).
PHRASES_GENERIQUES = [
    "Hé Julie", "Hé Marc", "Hé toi", "Hé voisin", "Hé les amis", "Hé attends", "Hé bonjour", "Hé regarde ça",
    "Eh bien", "Eh voyons", "Eh non", "Eh oui", "Eh ben dis donc", "Ok la lumière", "Ok Google", "Alexa",
    "Dis donc", "Dis-moi", "Allô", "Allô maman", "Bonjour tout le monde", "Bonsoir", "Bonne nuit", "Salut",
    "Merci beaucoup", "S'il te plaît", "Excusez-moi", "Pardon", "Attends une minute", "Viens ici", "Regarde ça",
    "Qu'est-ce que tu fais", "Où est-ce que tu vas", "Je ne sais pas", "C'est pas grave", "On y va", "Tout de suite",
    "Allume la lumière", "Éteins la télé", "Monte le volume", "Baisse le son", "Quelle heure est-il",
    "Quel temps fait-il demain", "Mets de la musique", "Arrête la musique", "Appelle ma mère", "Ouvre la porte",
    "Ferme la fenêtre", "Il fait beau", "Il pleut dehors", "Le nuage passe", "Un gros nuage gris", "Le soleil se lève",
    "Cumul de points", "Un cumul d'heures", "Le cumulonimbus", "Romulus et Rémus", "Un peu plus", "Plus ou moins",
    "Le souper est prêt", "À table", "Passe-moi le sel", "Je prendrais un café", "Tu veux du lait", "C'est délicieux",
    "Les enfants, dépêchez-vous", "Mets tes bottes", "Brosse-toi les dents", "Va te coucher", "Bonne journée",
    "Le hockey commence à sept heures", "Allez, les Canadiens", "Quel beau but", "Il neige encore", "Ça va bien",
    "Comment ça va", "Pas mal et toi", "À demain", "À bientôt", "Je reviens", "Ne bouge pas", "Écoute bien",
    "Combien ça coûte", "Je ne comprends pas", "Répète s'il te plaît", "C'est quoi ça", "Montre-moi", "Donne-moi ça",
    "Un deux trois", "Quatre cinq six", "Lundi mardi mercredi", "Janvier février mars", "La voiture est dans le garage",
    "J'ai oublié mes clés", "Où sont mes lunettes", "Le chat dort", "Le chien aboie", "La télé est trop forte",
    "On regarde un film", "Quel film veux-tu voir", "C'est l'heure", "Presque fini", "Voilà", "Bravo", "Super",
    "Hé Siri", "Hey Google", "Ok Maison", "Jarvis", "Computer", "Hé minus", "Hé Cumin", "Hé Lucas", "Hé Marius",
]


def phrases_generiques(fiche):
    """Les phrases génériques, plus les mots du mot demandé pris seuls (un mot à
    deux têtes ne doit pas se réveiller sur l'une des deux)."""
    cible = fiche["phrase"].strip().lower()
    seules = [m for m in fiche["phrase"].split() if len(m) > 1]
    toutes = [x for x in PHRASES_GENERIQUES if x.strip().lower() != cible] + seules
    return list(dict.fromkeys(toutes))


def _heures_ecrites(dossier):
    return sum(os.path.getsize(f) for f in Path(dossier).glob("*.wav")) / (2 * TAUX) / 3600


def _precharger_parole_plus(leger):
    """Trois morceaux de plus de MLS (train-00001..00003 sur 34 ; le 00000 sert
    déjà à apprendre, le 00033 à choisir) : 20 h au plus chacun, des
    locuteurs que ni la sélection ni le test n'ont entendus."""
    import pyarrow.parquet as pq

    for k in ((1,) if leger else (1, 2, 3)):
        d = CACHE / "parole_fr" / f"entrainement_plus{k}"
        if fait(d):
            continue
        fichier = f"train-{k:05d}-of-00034"
        p = telecharger(SOURCES["mls_fr"][0] + f"{fichier}.parquet", CACHE / "brut" / f"mls_fr_{fichier}.parquet")
        secondes = 0.0
        for i, a in enumerate(x for lot in pq.ParquetFile(p).iter_batches(columns=["audio"], batch_size=64)
                              for x in lot.column("audio").to_pylist()):
            if secondes >= (0.5 if leger else 20.0) * 3600:
                break
            try:
                son = lire_audio(a["bytes"])
                ecrire_wav(d / f"{i:06d}.wav", son)
                secondes += len(son) / TAUX
            except Exception as e:
                log.warning("mls plus %d ignoré : %s", i, e)
        marquer(d)
        p.unlink(missing_ok=True)


def _precharger_cv(leger):
    """Common Voice français (CC0) : train_0 pour apprendre (20 h au plus),
    dev pour choisir (4 h), test pour mesurer (8 h). Des milliers de locuteurs,
    dont des Québécois. Lu en flux dans les archives .tar du miroir."""
    import tarfile

    base = f"{SOURCES_LIBRES['common_voice_fr'][0]}/resolve/main/audio/fr"
    for partie, chemin, heures in (("entrainement", "train/fr_train_0.tar", 20.0), ("selection", "dev/fr_dev_0.tar", 4.0),
                                   ("test", "test/fr_test_0.tar", 8.0)):
        d = CACHE / "parole_cv" / partie
        if fait(d):
            continue
        secondes, n = 0.0, 0
        with urllib.request.urlopen(f"{base}/{chemin}", timeout=120) as r, tarfile.open(fileobj=r, mode="r|") as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith((".mp3", ".wav", ".flac")):
                    continue
                if secondes >= (0.3 if leger else heures) * 3600:
                    break
                try:
                    son = lire_audio(tar.extractfile(m).read())
                    ecrire_wav(d / f"{n:06d}.wav", son)
                    secondes += len(son) / TAUX
                    n += 1
                except Exception as e:
                    log.warning("cv %s ignoré : %s", m.name, e)
        if n == 0:
            raise RuntimeError(f"common voice {partie} : rien de lisible")
        log.info("common voice %s : %d clips, %.1f h", partie, n, secondes / 3600)
        marquer(d)


def _precharger_fleurs(leger):
    """FLEURS fr_fr (CC BY 4.0), parquet : l'entraînement et la validation
    ensemble, pour apprendre seulement (pas pour mesurer)."""
    import pyarrow.parquet as pq

    d = CACHE / "parole_fleurs" / "entrainement"
    if fait(d):
        return
    n = 0
    for partie in ("train", "validation"):
        p = telecharger(f"{HFM}/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/fr_fr/{partie}/0000.parquet",
                        CACHE / "brut" / f"fleurs_fr_{partie}.parquet")
        for lot in pq.ParquetFile(p).iter_batches(columns=["audio"], batch_size=32):
            for a in lot.column("audio").to_pylist():
                if leger and n >= 100:
                    break
                try:
                    ecrire_wav(d / f"{partie}-{n:06d}.wav", lire_audio(a["bytes"]))
                    n += 1
                except Exception as e:
                    log.warning("fleurs %d ignoré : %s", n, e)
    log.info("fleurs : %d clips", n)
    marquer(d)


DEMAND_SCENES = ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING", "OOFFICE",
                 "PCAFETER", "PRESTO", "PSTATION", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"]


def part_demand(scene):
    """Découpage PAR SCÈNE (les canaux d'une scène se ressemblent trop pour
    aller de part et d'autre) : trois scènes pour valider, trois pour tester."""
    i = DEMAND_SCENES.index(scene)
    return "test" if i % 6 == 5 else "validation" if i % 6 == 2 else "entrainement"


def _precharger_demand(leger):
    """DEMAND (CC BY 4.0, Zenodo) en 16 kHz : 3 canaux de 5 min par scène."""
    d = CACHE / "fond" / "demand"
    if fait(d):
        return
    for scene in (DEMAND_SCENES[:3] if leger else DEMAND_SCENES):
        z = telecharger(f"{SOURCES_LIBRES['demand'][0]}/files/{scene}_16k.zip?download=1", CACHE / "brut" / f"demand_{scene}_16k.zip")
        with zipfile.ZipFile(z) as zf:
            for nom in sorted(zf.namelist()):
                if nom.endswith(".wav") and Path(nom).stem in ("ch01", "ch02", "ch03"):
                    ecrire_wav(d / f"{scene}-{Path(nom).stem}.wav", lire_audio(zf.read(nom)))
    marquer(d)


def _precharger_musan(leger):
    """Les dossiers noise/ et music/ de MUSAN (CC BY 4.0), lus en flux dans
    l'archive de 11 Go. Bruit entier ; musique : 60 s au plus par morceau et
    400 morceaux au plus (une dizaine d'heures suffisent comme fond et comme
    son long). Tout en WAV 16 kHz dans fond/musan/<noise|music>-<nom>.wav."""
    import tarfile

    d = CACHE / "fond" / "musan"
    if fait(d):
        return
    limites = {"noise": 40 if leger else 10**9, "music": 20 if leger else 400}
    n = {"noise": 0, "music": 0}
    url = SOURCES_LIBRES["musan"][0]
    log.info("musan : lecture en flux de %s", url)
    with urllib.request.urlopen(url, timeout=120) as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            parts = Path(m.name).parts
            if not m.isfile() or len(parts) < 3 or parts[0] != "musan" or parts[1] not in n or not m.name.endswith(".wav"):
                continue
            genre = parts[1]
            if n[genre] >= limites[genre]:
                if all(n[g] >= limites[g] for g in n):
                    break
                continue
            try:
                son = lire_audio(tar.extractfile(m).read())
                if genre == "music":
                    son = son[: 60 * TAUX]
                ecrire_wav(d / f"{genre}-{Path(m.name).stem}.wav", son)
                n[genre] += 1
            except Exception as e:
                log.warning("musan %s ignoré : %s", m.name, e)
    if not all(n[g] for g in n):
        raise RuntimeError(f"musan : extraction incomplète {n}")
    log.info("musan : %s", n)
    marquer(d)


# --- 2. Synthèse Google fr-CA -----------------------------------------------------

GOOGLE = "https://texttospeech.googleapis.com/v1"


def google(methode, chemin, corps=None):
    url = f"{GOOGLE}/{chemin}{'&' if '?' in chemin else '?'}key={urllib.parse.quote(GOOGLE_CLE)}"
    req = urllib.request.Request(url, method=methode, data=json.dumps(corps).encode() if corps else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)




def formes(phrase, variantes):
    """La phrase et ses graphies, avec la ponctuation qui change l'intonation."""
    sortie = []
    for base in [phrase, *variantes]:
        mots_ = base.split()
        sortie += [base, base + ".", base + " !", base + " ?"]
        if len(mots_) > 1:
            sortie.append(mots_[0] + ", " + " ".join(mots_[1:]))
    return list(dict.fromkeys(sortie))


def synthetiser(texte, voix, vitesse, hauteur):
    """Le WAV d'une synthèse, depuis le cache si on l'a déjà payée."""
    cible = CACHE / "synthese" / "clips" / f"{empreinte(f'{voix}|{texte}|{vitesse}|{hauteur}', 20)}.wav"
    if cible.exists():
        return cible
    config = {"audioEncoding": "LINEAR16", "sampleRateHertz": TAUX, "speakingRate": vitesse}
    if hauteur:
        config["pitch"] = hauteur
    corps = {"input": {"text": texte}, "voice": {"languageCode": "fr-CA", "name": voix}, "audioConfig": config}
    try:
        r = google("POST", "text:synthesize", corps)
    except urllib.error.HTTPError as e:
        if e.code == 400 and hauteur:  # certaines voix (Chirp) refusent la hauteur
            return synthetiser(texte, voix, vitesse, 0)
        if e.code == 429:
            time.sleep(30)
            return synthetiser(texte, voix, vitesse, hauteur)
        log.warning("synthèse refusée (%s) : %s %r", e.code, voix, texte)
        return None
    import base64

    cible.parent.mkdir(parents=True, exist_ok=True)
    tmp = cible.with_suffix(".tmp")
    tmp.write_bytes(base64.b64decode(r["audioContent"]))
    os.replace(tmp, cible)
    return cible




def voix_fr_ca():
    cache = CACHE / "synthese" / "voix-fr-CA.json"
    voix = lire_json(cache)
    if not voix or time.time() - cache.stat().st_mtime > 7 * 86400:
        voix = sorted(v["name"] for v in google("GET", "voices?languageCode=fr-CA").get("voices", [])
                      if "fr-CA" in v.get("languageCodes", []))
        ecrire_json(cache, voix)
    return voix


def synthese_google(fiche, essai=False):
    """{"positif": [chemins], "negatif": [chemins], "voix": n, "caracteres": n}"""
    voix = voix_fr_ca()
    if essai:
        voix = voix[:4]
    pos, neg, caracteres = [], [], 0
    for v in voix:
        chirp = "Chirp" in v
        for texte in formes(fiche["phrase"], fiche.get("variantes", [])):
            for vitesse in (0.9, 1.0, 1.15):
                for hauteur in ((0,) if chirp else (-2.0, 0, 2.0)):
                    c = synthetiser(texte, v, vitesse, hauteur)
                    caracteres += len(texte)
                    if c:
                        pos.append(c)
        for texte in fiche.get("negatifs_proches", []):
            c = synthetiser(texte, v, 1.0, 0)
            caracteres += len(texte)
            if c:
                neg.append(c)
    log.info("synthèse Google : %d voix, %d positifs, %d négatifs proches", len(voix), len(pos), len(neg))
    return {"positif": sorted(set(pos)), "negatif": sorted(set(neg)), "voix": len(voix), "caracteres": caracteres}


# --- 2 bis. Synthèse libre (Piper) -------------------------------------------------

ASSISTANT_PIPER = r'''
import json, sys, wave
from piper import PiperVoice, SynthesisConfig
modele, travaux = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8"))
voix = PiperVoice.load(modele)
for t in travaux:
    cfg = SynthesisConfig(speaker_id=t["locuteur"], length_scale=t["longueur"], noise_scale=t["bruit"])
    with wave.open(t["sortie"], "wb") as f:
        voix.synthesize_wav(t["texte"], f, syn_config=cfg)
'''


def python_piper():
    """Un venv à part (piper-tts tire onnxruntime et sa propre pile) : on ne
    touche pas à celui de l'entraînement."""
    venv = CACHE / f"venv-piper-{PIPER_VERSION}"
    if not (venv / "bin" / "python").exists():
        tmp = Path(str(venv) + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", str(tmp)], check=True)
        subprocess.run([str(tmp / "bin" / "pip"), "install", "--quiet", "--no-cache-dir", f"piper-tts=={PIPER_VERSION}"], check=True)
        os.replace(tmp, venv)
    return venv / "bin" / "python"


def synthese_piper(fiche, essai=False):
    """Voix libres : mls (125 locuteurs) et siwis. Pas de hauteur (Piper n'en
    a pas) : trois vitesses, et un peu de variation de bruit. Mise en cache
    comme la synthèse Google."""
    py = python_piper()
    textes = formes(fiche["phrase"], fiche.get("variantes", []))
    proches = fiche.get("negatifs_proches", [])
    generiques = phrases_generiques(fiche) if not essai else phrases_generiques(fiche)[:10]
    pos, neg, gen, caracteres, nb_voix = [], [], [], 0, 0
    for nom, (url, n_loc, _) in PIPER_VOIX.items():
        modele = telecharger(url, CACHE / "synthese" / "piper" / Path(url).name)
        telecharger(url + ".json", CACHE / "synthese" / "piper" / (Path(url).name + ".json"))
        locuteurs = list(range(min(n_loc, 8 if essai else n_loc))) if n_loc > 1 else [None]
        nb_voix += len(locuteurs)
        travaux = []
        for loc in locuteurs:
            for texte in textes:
                for longueur in (0.87, 1.0, 1.15):  # ~vitesses 1,15 / 1,0 / 0,87
                    for bruit in ((0.667,) if n_loc > 1 else (0.667, 0.9, 1.1)):
                        travaux.append((texte, loc, longueur, bruit, True))
            for texte in proches:
                travaux.append((texte, loc, 1.0, 0.667, False))
            for texte in generiques:
                travaux.append((texte, loc, 1.0, 0.667, None))
        a_faire = []
        for texte, loc, longueur, bruit, positif in travaux:
            cible = CACHE / "synthese" / "clips" / f"{empreinte(f'piper-{nom}|{loc}|{texte}|{longueur}|{bruit}', 20)}.wav"
            caracteres += len(texte)
            (gen if positif is None else pos if positif else neg).append(cible)
            if not cible.exists():
                a_faire.append({"texte": texte, "locuteur": loc, "longueur": longueur, "bruit": bruit,
                                "sortie": str(cible) + ".piper.wav", "cible": str(cible)})
        if a_faire:
            cible_dir = CACHE / "synthese" / "clips"
            cible_dir.mkdir(parents=True, exist_ok=True)
            spec = CACHE / "synthese" / f"travaux-{nom}.json"
            ecrire_json(spec, a_faire)
            assistant = CACHE / "synthese" / "assistant_piper.py"
            assistant.write_text(ASSISTANT_PIPER, encoding="utf-8")
            log.info("piper %s : %d clips à synthétiser", nom, len(a_faire))
            subprocess.run([str(py), str(assistant), str(modele), str(spec)], check=True)
            for t in a_faire:
                brut = Path(t["sortie"])
                ecrire_wav(t["cible"], lire_audio(brut))
                brut.unlink()
            spec.unlink(missing_ok=True)
    log.info("synthèse libre : %d voix, %d positifs, %d négatifs proches, %d négatifs génériques",
             nb_voix, len(pos), len(neg), len(gen))
    return {"positif": sorted(set(pos)), "negatif": sorted(set(neg)), "generiques": sorted(set(gen)), "voix": nb_voix,
            "caracteres": caracteres, "moteur": "piper"}


def synthese_publique(fiche, essai=False):
    """Voix libres toujours ; les voix Google seulement en OPTION (clé présente
    ET EVEIL_PUBLIC_GOOGLE=1) : leurs conditions ne sont pas tranchées pour un
    modèle distribué."""
    libre = synthese_piper(fiche, essai)
    if not (GOOGLE_CLE and os.environ.get("EVEIL_PUBLIC_GOOGLE") == "1"):
        return libre
    g = synthese_google(fiche, essai)
    # Les mêmes phrases génériques, dites par les voix Google (les positifs
    # déjà synthétisés viennent du cache : seule la liste de négatifs change).
    gg = synthese_google(dict(fiche, variantes=[], negatifs_proches=phrases_generiques(fiche)[:10 if essai else None]), essai)
    return {"positif": sorted(set(libre["positif"]) | set(g["positif"])),
            "negatif": sorted(set(libre["negatif"]) | set(g["negatif"])),
            "generiques": sorted(set(libre["generiques"]) | set(gg["negatif"])),
            "voix": libre["voix"] + g["voix"], "caracteres": libre["caracteres"] + g["caracteres"] + gg["caracteres"],
            "moteur": "piper+google"}



def separer(chemins):
    s = {"entrainement": [], "validation": [], "test": []}
    for p in chemins:
        s[cote(p)].append(p)
    return s


# --- 4. Caractéristiques (spectrogrammes microWakeWord) --------------------------


class MesClips:
    """Ce que SpectrogramGeneration attend d'un objet Clips : un générateur
    audio_generator(split, repeat). On évite microwakeword.audio.clips, qui
    tire la bibliothèque datasets et son piège torchcodec (juillet 2026)."""

    def __init__(self, parties, duree_min_s=None):
        self.parties = parties  # {"train": [...], "validation": [...], "test": [...]}
        self.duree_min_s = duree_min_s

    def audio_generator(self, split=None, repeat=1):
        chemins = self.parties.get(split, []) if split else [p for v in self.parties.values() for p in v]
        for _ in range(repeat):
            for p in chemins:
                try:
                    son = lire_audio(p)
                except Exception as e:
                    log.warning("%s illisible : %s", p, e)
                    continue
                if self.duree_min_s:
                    n = int(self.duree_min_s * TAUX)
                    while len(son) < n:
                        son = np.concatenate([son, son])
                yield son


def sans_clips():
    """microwakeword.audio.spectrograms importe microwakeword.audio.clips, qui
    importe audio_metadata et datasets pour la seule classe Clips, que l'on
    n'utilise pas (MesClips la remplace). Deux modules vides suffisent : on
    n'installe ni datasets ni son piège torchcodec."""
    import types

    for nom in ("audio_metadata", "datasets"):
        if nom not in sys.modules:
            try:
                __import__(nom)
            except ImportError:
                sys.modules[nom] = types.ModuleType(nom)


def augmenteur(fonds, rir):
    sans_clips()
    from microwakeword.audio.augmentation import Augmentation

    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.1,
            "BandStopFilter": 0.1, "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75,
            "Gain": 1.0, "RIR": 0.5,
        },
        impulse_paths=[str(rir)],
        background_paths=[str(f) for f in fonds],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def ecrire_mmap(dossier, generateur):
    from mmap_ninja.ragged import RaggedMmap

    dossier = Path(dossier)
    if dossier.exists():
        shutil.rmtree(dossier)
    dossier.parent.mkdir(parents=True, exist_ok=True)
    n = {"n": 0}

    def compter():
        for x in generateur:
            # Un clip trop court (ou vide) donne un spectrogramme sans image :
            # microWakeWord plante à l'entraînement (np.pad sur une forme (0,),
            # essai public du 2026-11-12). On l'écarte et on le dit.
            if getattr(x, "ndim", 2) != 2 or len(x) == 0:
                n["ecartes"] = n.get("ecartes", 0) + 1
                continue
            n["n"] += 1
            yield x

    RaggedMmap.from_generator(out_dir=str(dossier), sample_generator=compter(), batch_size=100, verbose=False)
    if n.get("ecartes"):
        log.warning("%s : %d spectrogrammes vides écartés", dossier.parent.name, n["ecartes"])
    return n["n"]


def jeu(racine, nom, parties, aug, repetitions=(2, 1, 1), ambiant=False):
    """Écrit training/validation/testing de ce jeu ; rend le compte par partie.
    ambiant : du son long sans augmentation, découpé pour mesurer les
    fausses détections (validation_ambient / testing_ambient)."""
    sans_clips()
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    comptes = {}
    clips = MesClips(parties)
    for partie, cle, rep in (("training", "train", repetitions[0]), ("validation", "validation", repetitions[1]),
                             ("testing", "test", repetitions[2])):
        if not parties.get(cle):
            continue
        if ambiant and partie != "training":
            gen = SpectrogramGeneration(clips=clips, augmenter=None, step_ms=10, split_spectrogram_duration_s=3.0)
            sortie = Path(racine) / nom / f"{partie}_ambient" / f"{nom}_mmap"
        elif aug is None:
            # Son long, sans augmentation : le spectrogramme entier, que
            # l'entraînement tronque au hasard (truncation_strategy random).
            gen = SpectrogramGeneration(clips=clips, augmenter=None, step_ms=10)
            sortie = Path(racine) / nom / partie / f"{nom}_mmap"
        else:
            glisse = 10 if partie != "testing" else 1
            gen = SpectrogramGeneration(clips=clips, augmenter=aug, step_ms=10, slide_frames=glisse)
            sortie = Path(racine) / nom / partie / f"{nom}_mmap"
        comptes[partie] = ecrire_mmap(sortie, gen.spectrogram_generator(split=cle, repeat=rep))
    log.info("caractéristiques %s : %s", nom, comptes)
    return comptes


def parties_de(chemins):
    s = separer(chemins)
    return {"train": s["entrainement"], "validation": s["validation"], "test": s["test"]}





# --- 6. L'entraînement ------------------------------------------------------------

LANCEUR = """
import os, runpy, sys, types
for nom in ("audio_metadata", "datasets"):  # voir sans_clips() dans entrainer.py
    try:
        __import__(nom)
    except ImportError:
        sys.modules[nom] = types.ModuleType(nom)
import tensorflow as tf
limite = int(os.environ.get("EVEIL_TF_MEMOIRE_MIO", "0"))
for g in tf.config.list_physical_devices("GPU"):
    if limite:
        tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=limite)])
# La validation de microWakeWord passe tout un jeu (numpy) à model.evaluate :
# Keras le copie d'un bloc sur la carte, et avec le 27B à côté il n'y a que
# 3 à 4 Go (« Dst tensor is not initialized », 2026-11-04). Même calcul, par
# lots, depuis le processeur : le jeu reste en mémoire vive, seuls les lots
# vont sur la carte. PAS from_generator : TensorFlow garde chaque générateur
# dans un registre global, avec le jeu entier, et une copie de plus restait à
# chaque validation (OOMKilled à 20 Gio au pas 2000, 2026-11-04).
import numpy as np, keras
_evaluer = keras.Model.evaluate
def _evaluer_par_lots(self, x=None, y=None, batch_size=None, **kw):
    if isinstance(x, np.ndarray):
        with tf.device("/CPU:0"):
            lots = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size or 32)
        return _evaluer(self, lots, **kw)
    return _evaluer(self, x, y, batch_size=batch_size, **kw)
keras.Model.evaluate = _evaluer_par_lots
sys.argv = ["model_train_eval"] + sys.argv[1:]
runpy.run_module("microwakeword.model_train_eval", run_name="__main__")
"""


# Les variantes essayées à chaque entraînement ; la meilleure sur la sélection
# (jamais sur le test) est livrée. Les spectrogrammes sont faits une fois.
#   base        : la recette ;
#   prudent     : les négatifs comptent deux fois plus (moins de fausses détections).
VARIANTES = [
    {"nom": "base", "poids_negatif": 20},
    {"nom": "prudent", "poids_negatif": 40},
]
PAS_DEFAUT = 50000


def phases(pas):
    """Trois paliers : apprendre (60 %), affiner (30 %), polir sans masquage
    (10 %). 10 000 pas à taux constant oscillaient de 3 à 300 fausses
    détections par heure d'une validation à l'autre (2026-11-04)."""
    a, b = int(pas * 0.6), int(pas * 0.3)
    return [a, b, pas - a - b]


def entrainer_modele(travail, jeux, pas, carte, memoire_mio, variante):
    config = {
        "window_step_ms": 10,
        "train_dir": str(travail / f"modele-{variante['nom']}"),
        "features": jeux,
        "training_steps": phases(pas),
        "positive_class_weight": [1],
        "negative_class_weight": [variante["poids_negatif"]],
        "learning_rates": [0.001, 0.0005, 0.0001],
        "batch_size": 128,
        # Masquage du spectrogramme (SpecAugment).
        "time_mask_max_size": [5, 5, 0], "time_mask_count": [2, 2, 0],
        "freq_mask_max_size": [5, 5, 0], "freq_mask_count": [2, 2, 0],
        "eval_step_interval": max(100, pas // 100),
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    import yaml

    params = travail / f"training_parameters-{variante['nom']}.yaml"
    params.write_text(yaml.safe_dump(config, allow_unicode=True))
    (travail / "lanceur.py").write_text(LANCEUR)
    env = dict(os.environ, EVEIL_TF_MEMOIRE_MIO=str(memoire_mio),
               CUDA_VISIBLE_DEVICES="" if carte is None else str(carte))
    cmd = [sys.executable, str(travail / "lanceur.py"),
           "--training_config", str(params),
           "--train", "1", "--restore_checkpoint", "1",
           "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0",
           "--test_tflite_nonstreaming_quantized", "0", "--test_tflite_streaming", "0",
           "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
           "mixednet", "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1, 1, 1, 1",
           "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]", "--residual_connection", "0,0,0,0",
           "--first_conv_filters", "32", "--first_conv_kernel_size", "5", "--stride", "3"]
    debut = time.monotonic()
    subprocess.run(cmd, check=True, env=env, cwd=str(travail))
    tflite = Path(config["train_dir"]) / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if not tflite.exists():
        raise RuntimeError("l'entraînement n'a pas produit de modèle")
    return tflite, (time.monotonic() - debut) / 3600


# --- 7. L'évaluation, sur le son brut, comme l'appareil ---------------------------


# Le seuil EST celui de l'appareil. ESPHome (micro_wake_word, 2026.8) garde les
# N dernières sorties du modèle en octets (0 à 255) et déclenche quand leur
# SOMME dépasse cran × N, cran = int(probability_cutoff × 255) :
#   detected = sum > probability_cutoff_ * sliding_window_size_
# Les crans au-dessus de 0,99 existent donc : 253 (≈ 0,992) et 254 (≈ 0,996) ;
# 255 ne déclencherait jamais. Et la fenêtre N est un second levier : une
# pointe brève sur de la parole ne remplit pas 8 sorties de suite, un vrai
# mot d'éveil si. On mesure toute la grille (N, cran) et on choisit.
FENETRES = (5, 6, 7, 8, 10)
CRANS = np.arange(128, 255)  # 0,50 à 0,996 ; jamais plus bas
PAS_S = 0.03  # une inférence toutes les 30 ms (stride 3 × 10 ms)


def seuil_du_cran(cran):
    """Ce que l'appareil affiche (probability_cutoff_ / 255)."""
    return round(int(cran) / 255, 3)


def seuil_pour_manifeste(cran):
    """Le flottant dont int(× 255) redonne exactement ce cran."""
    return round((int(cran) + 0.5) / 255, 4)


def manifeste(mot, phrase, cran, fichier, fenetre=5):
    # Pas de « website » vide : ESPHome 2026.8 le lit comme une URL et refuse
    # tout le firmware (« Expected a file scheme or a URL scheme with host »).
    return {
        "type": "micro",
        "wake_word": phrase,
        "author": "mots-eveil (voix de synthèse libres, données libres)",
        "model": fichier,
        "trained_languages": ["fr"],
        "version": 2,
        "micro": {
            "probability_cutoff": seuil_pour_manifeste(cran),
            "feature_step_size": 10,
            "sliding_window_size": int(fenetre),
            "tensor_arena_size": 40000,
            "minimum_esphome_version": "2024.7.0",
        },
    }


def reglage_du_manifeste(json_modele):
    micro = lire_json(json_modele)["micro"]
    return int(micro.get("sliding_window_size", 5)), int(float(micro["probability_cutoff"]) * 255)


class Ecouteur:
    """Les sorties brutes d'un modèle sur un son, en octets comme sur
    l'appareil, par pymicro-wakeword."""

    def __init__(self, json_modele):
        from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures

        self.Features = MicroWakeWordFeatures
        self.modele = MicroWakeWord.from_config(Path(json_modele))

    def octets(self, son):
        pcm = (np.clip(son, -1, 1) * 32767).astype(np.int16).tobytes()
        self.modele.reset()
        brut = []
        for f in self.Features().process_streaming(pcm):
            p = self.modele.process_streaming_prob(f)
            if p is None or (p == 0.0 and len(self.modele._features) != 0):
                continue
            brut.append(p)
        echelle = float(getattr(self.modele, "output_scale", 1 / 256)) or 1 / 256
        zero = int(getattr(self.modele, "output_zero_point", 0))
        return np.clip(np.rint(np.asarray(brut, dtype=np.float64) / echelle) + zero, 0, 255).astype(np.int32)


def sommes(octets, fenetre):
    if len(octets) < fenetre:
        return np.zeros(1, dtype=np.int64)
    return np.convolve(octets, np.ones(fenetre, dtype=np.int64), mode="valid")


def compter_par_cran(s, fenetre):
    """Pour chaque cran : les déclenchements, avec la période réfractaire."""
    attente = int(FENETRE_ARRET_S / PAS_S)
    idx = np.flatnonzero(s > CRANS[0] * fenetre)
    vals = s[idx]
    comptes = np.zeros(len(CRANS), dtype=int)
    for k, cran in enumerate(CRANS):
        sel = idx[vals > cran * fenetre]
        if not len(sel):
            break  # plus haut, encore moins
        n, dernier = 0, -10 ** 9
        for i in sel:
            if i - dernier > attente:
                n += 1
                dernier = i
        comptes[k] = n
    return comptes


def mesurer(json_modele, eval_, fenetres=FENETRES):
    """Pour un modèle et chaque fenêtre : la plus haute somme de chaque
    positif, et les déclenchements sur le son long à chaque cran."""
    e = Ecouteur(json_modele)
    f_ = [str(n) for n in fenetres]
    sortie = {"fenetres": [int(n) for n in fenetres], "positifs": {}, "proches": {n: [] for n in f_}, "long": {}}
    for groupe, chemins in eval_["positifs"].items():
        sortie["positifs"][groupe] = {n: [] for n in f_}
        for p in chemins:
            o = e.octets(lire_audio(p))
            for n in fenetres:
                sortie["positifs"][groupe][str(n)].append(int(sommes(o, n).max()))
    for p in eval_["proches"]:
        o = e.octets(lire_audio(p))
        for n in fenetres:
            sortie["proches"][str(n)].append(int(sommes(o, n).max()))
    for groupe, chemins in eval_["long"].items():
        heures, comptes = 0.0, {n: np.zeros(len(CRANS), dtype=int) for n in f_}
        for p in chemins:
            son = lire_audio(p)
            heures += len(son) / TAUX / 3600
            o = e.octets(son)
            for n in fenetres:
                comptes[str(n)] += compter_par_cran(sommes(o, n), n)
        sortie["long"][groupe] = {"heures": heures, "comptes": {n: c.tolist() for n, c in comptes.items()}}
    return sortie


def resumer(m, fenetre, cran):
    f, k = str(fenetre), int(np.flatnonzero(CRANS == cran)[0])
    seuil = cran * int(fenetre)
    rate = {g: (sum(1 for x in v[f] if x <= seuil), len(v[f])) for g, v in m["positifs"].items()}
    longs = {g: (d["comptes"][f][k], d["heures"]) for g, d in m["long"].items()}
    tot_n = sum(n for n, _ in longs.values())
    tot_h = sum(h for _, h in longs.values())
    proches = (sum(1 for x in m["proches"][f] if x > seuil), len(m["proches"][f]))
    return {"fenetre": int(fenetre), "cran": int(cran), "seuil": seuil_du_cran(cran), "rates": rate, "longs": longs,
            "fa_h": (tot_n / tot_h) if tot_h else None, "proches": proches}


def _long(m, fenetre):
    heures = sum(d["heures"] for d in m["long"].values())
    comptes = np.sum([d["comptes"][str(fenetre)] for d in m["long"].values()], axis=0) if m["long"] \
        else np.zeros(len(CRANS), dtype=int)
    return heures, comptes


def choisir_seuils(m, cible, fenetre):
    """Le cran, pour la fenêtre choisie : le plus bas (le plus de mots
    entendus) dont la borne haute des fausses détections tient la cible. La
    borne est prudente : (n + 3) / heures, puisque 0 détection en H heures ne
    prouve qu'environ 3/H par heure (95 %). Sans preuve possible, le cran le
    plus haut (254), et le rapport le dit. Les trois sensibilités du Voice
    PE : la moitié de la cible (peu sensible), la cible, le double (très)."""
    heures, comptes = _long(m, fenetre)

    def plus_bas(fa_max):
        for cran, n in zip(CRANS, comptes):
            if heures and (n + 3) / heures <= fa_max:
                return int(cran), True
        return int(CRANS[-1]), False

    principal, prouve = plus_bas(cible)
    peu, _ = plus_bas(cible / 2)
    tres, _ = plus_bas(2 * cible)
    return {"fenetre": int(fenetre), "principal": principal, "peu": max(peu, principal), "moyen": principal,
            "tres": min(tres, principal), "heures": heures, "prouve": prouve,
            "heures_pour_prouver": 3 / cible if cible else None}


def meilleur_par_fenetre(m, cible):
    """Pour chaque fenêtre : le cran le plus bas dont les fausses détections
    MESURÉES tiennent la cible, et ce qu'il entend. D'abord
    la synthèse entendue, puis moins de fausses détections, puis la plus petite
    fenêtre (plus rapide)."""
    sortie = []
    for fenetre in m["fenetres"]:
        heures, comptes = _long(m, fenetre)
        cran = int(CRANS[-1])
        for c, n in zip(CRANS, comptes):
            if heures and n / heures <= cible:
                cran = int(c)
                break
        r = resumer(m, fenetre, cran)
        reels = [1 - n / t for g, (n, t) in r["rates"].items() if g != "synthèse" and t]
        synth = [1 - n / t for g, (n, t) in r["rates"].items() if g == "synthèse" and t]
        note = (float(np.mean(reels)) if reels else 0.0, float(np.mean(synth)) if synth else 0.0,
                -(r["fa_h"] or 0.0), -int(fenetre))
        sortie.append({"fenetre": int(fenetre), "cran": cran, "seuil": seuil_du_cran(cran), "note": note,
                       "entendus_reels": note[0], "entendus_synthese": note[1], "fa_h": r["fa_h"], "heures": heures,
                       "rates": r["rates"]})
    return sortie


def note_selection(m, cible):
    """Pour choisir une variante et sa fenêtre, sur la sélection : le meilleur
    réglage de la grille (la borne prudente n'y servirait pas : 6 h ne
    suffiraient pas, et toutes les variantes sont jugées pareil)."""
    grille = meilleur_par_fenetre(m, cible)
    meilleur = max(grille, key=lambda x: x["note"])
    return dict(meilleur, grille=[{k: v for k, v in g.items() if k != "rates"} for g in grille])


# --- 8. Le rapport et la livraison ------------------------------------------------


def pct(n, total):
    return f"{100 * n / total:.0f} %" if total else "—"


def ligne_resume(nom, r):
    rates = " · ".join(f"{g} {pct(n, t)} ({n}/{t})" for g, (n, t) in sorted(r["rates"].items()))
    fa = "—" if r["fa_h"] is None else f"{r['fa_h']:.2f}"
    return (f"| {nom} | {r['seuil']:.3f} (cran {r['cran']}, fenêtre {r['fenetre']}) | {rates or '—'} | {fa} | "
            f"{r['proches'][0]}/{r['proches'][1]} |")


def yaml_voice_pe(mot, seuils, url_json):
    s = seuil_du_cran
    return f"""# Le mot d'éveil {mot}, à AJOUTER aux mots officiels (sans !remove : la
# sélection « Wake word » de HA reste valide).
# Crans tels que l'appareil les compare : déclenche si la somme des
# {seuils['fenetre']} dernières sorties (octets) dépasse cran × {seuils['fenetre']}.
micro_wake_word:
  models:
    - model: {url_json}
      id: {mot}
      sliding_window_size: {seuils['fenetre']}

select:
  - id: !extend wake_word_sensitivity   # sa lambda nomme les mots officiels : on ajoute le nôtre
    on_value:
      - lambda: |-
          if (x == "Slightly sensitive") {{ id({mot}).set_probability_cutoff({seuils['peu']}); }}   // {s(seuils['peu']):.3f}
          else if (x == "Moderately sensitive") {{ id({mot}).set_probability_cutoff({seuils['moyen']}); }}   // {s(seuils['moyen']):.3f}
          else if (x == "Very sensitive") {{ id({mot}).set_probability_cutoff({seuils['tres']}); }}   // {s(seuils['tres']):.3f}
"""


def tableau_grille(grille, titre):
    lignes = ["", f"## {titre}", "",
              "Pour chaque fenêtre, le cran le plus bas dont les fausses détections mesurées tiennent la cible.", "",
              "| Fenêtre | Cran | Seuil | Vraies prises entendues | Synthèse entendue | Fausses détections / h |",
              "|---|---|---|---|---|---|"]
    for g in grille:
        fa = "—" if g["fa_h"] is None else f"{g['fa_h']:.2f}"
        lignes.append(f"| {g['fenetre']} | {g['cran']} | {g['seuil']:.3f} | {pct(g['entendus_reels'], 1)} | "
                      f"{pct(g['entendus_synthese'], 1)} | {fa} |")
    return lignes


def rapport_public_md(mot, fiche, version, synth, comptes, seuils, neuf, heures_gpu, sources, variantes=()):
    """Le rapport d'un modèle public : ce qui est mesuré, et surtout ce qui ne l'est PAS."""
    cible = fiche.get("cible_fausses_detections_h", 0.5)
    lignes = [
        f"# {fiche['phrase']} — modèle de mot d'éveil, entraînement {version}",
        "",
        "**Lis ceci d'abord.** Ce modèle a été entraîné **uniquement sur des voix de synthèse** et des données libres, "
        "sans une seule vraie voix humaine disant le mot. Les chiffres ci-dessous ne disent donc PAS comment il se "
        "comporte avec ta voix, ta pièce et ton micro : teste-le chez toi, et garde l'ancien réglage sous la main.",
        "",
        "## Ce qui est mesuré",
        "",
        f"Cible : **{cible} fausse détection par heure** au plus, mesurée sur du son long LIBRE : parole française lue "
        "(MLS) et bruits / musique (MUSAN), jamais entendus à l'entraînement.",
        "",
        "| Mesure | Résultat |",
        "|---|---|",
        f"| Fenêtre glissante / cran choisi | {seuils['fenetre']} sorties / {seuils['principal']} (seuil {seuil_du_cran(seuils['principal']):.3f}) |",
        f"| Voix de synthèse gardées de côté, entendues | {pct(*neuf['rates']['synthèse'])} |" if "synthèse" in neuf["rates"] else "| Voix de synthèse gardées de côté | — |",
        f"| Fausses détections par heure (son long libre) | {'—' if neuf['fa_h'] is None else format(neuf['fa_h'], '.2f')} |",
        f"| Heures de son long mesurées | {seuils['heures']:.2f} h ({', '.join(f'{g} {h:.2f} h' for g, (_, h) in neuf['longs'].items())}) |",
        f"| Phrases proches qui déclenchent | {neuf['proches'][0]}/{neuf['proches'][1]} |",
        "",
    ]
    if neuf["fa_h"] is not None and neuf["fa_h"] > cible:
        lignes += [f"**La cible n'est PAS tenue** : {neuf['fa_h']:.2f} fausse détection par heure, même au cran le plus "
                   f"prudent ({cible} visée). Ne pas installer ce modèle tel quel : plus de négatifs libres, ou une "
                   "variante plus prudente, avant de s'en servir.", ""]
    elif not seuils["prouve"]:
        lignes += [f"**Attention : pas assez d'heures pour prouver la cible** (il en faut au moins {seuils['heures_pour_prouver']:.0f} "
                   "sans fausse détection). Le seuil est mis au plus prudent.", ""]
    lignes += [
        "## Ce qui n'est PAS mesuré",
        "",
        "- Aucune vraie voix du mot : le taux « entendu » vient de la même famille de voix de synthèse que "
        "l'entraînement (voix différentes d'un côté et de l'autre du partage, mais toutes synthétiques). Il est "
        "optimiste.",
        "- Aucune ambiance de maison (télé, repas, enfants) ni ton micro : les fausses détections réelles seront "
        "probablement plus nombreuses que ci-dessus.",
        "- Voix françaises de France (Piper mls / siwis), pas du français québécois : l'accent n'est pas couvert "
        "sauf si l'option Google fr-CA a été activée (voir les sources).",
        "",
        "## Variantes essayées",
        "",
    ]
    for v in variantes:
        sv = v["selection"]
        lignes.append(f"- {v['nom']}{' (livrée)' if v.get('livree') else ''} : fenêtre {sv['fenetre']}, seuil {sv['seuil']:.3f}, "
                      f"{v['heures_gpu']:.2f} h-GPU")
    lignes += [
        "",
        "## Ce qui a servi",
        "",
        f"- Synthèse : moteur {synth.get('moteur', '?')}, {synth['voix']} voix, {len(synth['positif'])} prises du mot, "
        f"{len(synth['negatif'])} phrases proches.",
        "- Aucune vraie prise, aucune ambiance de la maison, aucun réveil d'appareil.",
        f"- Spectrogrammes : {json.dumps(comptes, ensure_ascii=False)}",
        f"- Heures-GPU : {heures_gpu:.2f}",
        "",
        "## Sources et licences",
        "",
    ]
    lignes += [f"- {x['nom']} : {x['role']} — {x['licence']}" for x in sources]
    lignes += ["", "Écarté exprès de ce mode : AudioSet (audio YouTube), FMA (licences NC) et les négatifs officiels de "
               "microWakeWord (cc-by-nc-4.0)."]
    return "\n".join(lignes) + "\n"




# --- Le tout ----------------------------------------------------------------------


def jeux_eval(synth, amb, essai=False):
    """La sélection (prises de synthèse de validation, 6 h de parole, fond libre
    de validation) pour choisir ; le test (prises de test, ~20 h de parole,
    fond libre de test) pour mesurer et fixer le cran."""
    def jeu(partie, long):
        e = {
            "positifs": {"synthèse": [p for p in synth["positif"] if cote(p) == partie][:300]},
            "proches": [p for p in synth["negatif"] if cote(p) == partie],
            "long": long,
        }
        e["positifs"] = {k: v for k, v in e["positifs"].items() if v}
        e["long"] = {k: v for k, v in e["long"].items() if v}
        return e

    pf = lambda partie: sorted((CACHE / "parole_fr" / partie).glob("*.wav"))
    cv = lambda partie: sorted((CACHE / "parole_cv" / partie).glob("*.wav"))
    longs_sel = {"fond_libre": amb["validation"], "parole_fr": pf("eval")[:30] if essai else pf("selection")}
    longs_test = {"fond_libre": amb["test"], "parole_fr": pf("eval")[:60] if essai else pf("eval") + pf("test")}
    # Common Voice (dont des Québécois) mesure en plus, jamais entendu à l'apprentissage.
    longs_sel["parole_cv"], longs_test["parole_cv"] = cv("selection"), cv("test")
    return jeu("validation", longs_sel), jeu("test", longs_test)


def choisir_carte(memoire_mio):
    """L'index de la carte NVIDIA qui a le plus de mémoire libre (EVEIL_GPU le
    force), ou une erreur s'il n'y en a pas assez. Aucune file d'attente : si
    la carte sert à autre chose, libère-la avant de lancer l'entraînement."""
    if os.environ.get("EVEIL_GPU", "").strip():
        return int(os.environ["EVEIL_GPU"])
    try:
        sortie = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
        cartes = [tuple(int(x) for x in ligne.split(",")) for ligne in sortie.strip().splitlines()]
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        raise RuntimeError(f"nvidia-smi : {e} (sans carte graphique, utiliser --sans-gpu)")
    index, libre = max(cartes, key=lambda c: c[1])
    if libre < memoire_mio + 512:
        raise RuntimeError(f"pas assez de mémoire libre sur la carte {index} ({libre} Mio, il en faut {memoire_mio + 512})")
    log.info("GPU : carte %d (%d Mio libres)", index, libre)
    return index


def entrainer_mot(mot, essai=False, pas=None, sans_gpu=False):
    garde_public()
    fiche = lire_json(DONNEES / "mots" / mot / "fiche.json")
    if not fiche:
        raise RuntimeError(f"pas de fiche pour {mot}")
    version = dt.datetime.now().strftime("%Y%m%d-%H%M")
    travail = CACHE / "travail" / mot / version
    travail.mkdir(parents=True, exist_ok=True)
    log.info("== %s (%s), version %s", mot, fiche["phrase"], version)
    try:
        return _entrainer_mot(mot, fiche, version, travail, essai, pas, sans_gpu)
    finally:
        # Le travail contient des spectrogrammes : effacé même après un échec
        # (EVEIL_GARDER_TRAVAIL=1 seulement pour déboguer).
        if os.environ.get("EVEIL_GARDER_TRAVAIL") != "1":
            shutil.rmtree(travail.parent, ignore_errors=True)


def _entrainer_mot(mot, fiche, version, travail, essai, pas, sans_gpu):
    precharger_public(leger=essai)
    synth = synthese_publique(fiche, essai=essai)
    if not synth["positif"]:
        raise RuntimeError("aucune synthèse : rien à apprendre")

    # Le « son long » : bruits et musique libres (MUSAN, DEMAND), coupés comme
    # les prises de synthèse (même empreinte stable). Les fonds de
    # l'augmentation ne prennent que la part d'entraînement (liens), pas la
    # validation ni le test.
    amb = separer(sorted((CACHE / "fond" / "musan").glob("*.wav")))
    for p in sorted((CACHE / "fond" / "demand").glob("*.wav")):
        amb[part_demand(p.name.split("-")[0])].append(p)
    d = travail / "fond_libre"
    d.mkdir(exist_ok=True)
    for p in amb["entrainement"]:
        (d / p.name).symlink_to(p)
    fonds = [d]
    log.info("fond libre : %s", {k: len(v) for k, v in amb.items()})
    aug = augmenteur(fonds, CACHE / "rir")

    feat = travail / "caracteristiques"
    comptes, jeux = {}, []

    def ajouter(nom, chemins_ou_parties, verite, poids, truncation, aug_=aug, reps=(2, 1, 1), ambiant=False):
        parties = chemins_ou_parties if isinstance(chemins_ou_parties, dict) else parties_de(chemins_ou_parties)
        if not any(parties.values()):
            return
        comptes[nom] = jeu(feat, nom, parties, aug_, reps, ambiant)
        jeux.append({"features_dir": str(feat / nom), "sampling_weight": poids, "penalty_weight": 1.0,
                     "truth": verite, "truncation_strategy": truncation, "type": "mmap"})

    ajouter("positifs_synthese", synth["positif"], True, 2.0, "truncate_start")
    ajouter("proches", synth["negatif"], False, 3.0, "truncate_start")
    # Les phrases génériques, dites par les mêmes voix que les positifs.
    ajouter("negatifs_synthese", synth.get("generiques", []), False, 4.0, "truncate_start")
    ajouter("fond_libre", {"train": amb["entrainement"], "validation": amb["validation"]}, False, 8.0,
            "random", aug_=None, reps=(1, 1, 1), ambiant=True)
    parole = sorted(f for nom in ("entrainement", "entrainement_plus", "entrainement_plus1", "entrainement_plus2",
                                  "entrainement_plus3") for f in (CACHE / "parole_fr" / nom).glob("*.wav"))
    parole += sorted((CACHE / "parole_cv" / "entrainement").glob("*.wav"))
    parole += sorted((CACHE / "parole_fleurs" / "entrainement").glob("*.wav"))
    log.info("parole d'apprentissage : %d clips", len(parole))
    ajouter("parole_fr", {"train": parole}, False, 5.0, "random", aug_=None, reps=(1, 1, 1))

    cible = float(fiche.get("cible_fausses_detections_h", 0.5))
    selection, eval_ = jeux_eval(synth, amb, essai)

    # Les variantes, sur la même carte, l'une après l'autre ; la meilleure sur
    # la sélection est livrée.
    variantes = VARIANTES[:1] if essai else VARIANTES[:2]
    pas = pas or (300 if essai else PAS_DEFAUT)
    memoire = int(os.environ.get("EVEIL_TF_MEMOIRE_MIO", "3000"))
    carte = None if sans_gpu else choisir_carte(memoire)
    faites, heures_gpu = [], 0.0
    for v in variantes:
        tfl, h = entrainer_modele(travail, jeux, pas, carte, memoire, v)
        heures_gpu += h
        j = travail / f"candidat-{v['nom']}.json"
        shutil.copyfile(tfl, travail / f"candidat-{v['nom']}.tflite")
        ecrire_json(j, manifeste(mot, fiche["phrase"], CRANS[0], f"candidat-{v['nom']}.tflite"))
        faites.append(dict(v, heures_gpu=h, tflite=travail / f"candidat-{v['nom']}.tflite",
                           selection=note_selection(mesurer(j, selection), cible)))
        log.info("variante %s : %s", v["nom"], faites[-1]["selection"])
    choisie = max(faites, key=lambda v: v["selection"]["note"])
    choisie["livree"] = True
    fenetre = choisie["selection"]["fenetre"]
    log.info("variante livrée : %s, fenêtre %d", choisie["nom"], fenetre)

    # Évaluation du modèle choisi sur les mêmes sons (eval_).
    sortie = DONNEES / "modeles" / mot / version
    sortie.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(choisie["tflite"], sortie / f"{mot}.tflite")
    ecrire_json(sortie / f"{mot}.json", manifeste(mot, fiche["phrase"], CRANS[0], f"{mot}.tflite"))
    m_neuf = mesurer(sortie / f"{mot}.json", eval_, (fenetre,))
    seuils = choisir_seuils(m_neuf, cible, fenetre)
    ecrire_json(sortie / f"{mot}.json", manifeste(mot, fiche["phrase"], seuils["principal"], f"{mot}.tflite", fenetre))
    neuf = resumer(m_neuf, fenetre, seuils["principal"])

    sources = [{"nom": k, "role": v[2], "licence": v[1]} for k, v in SOURCES_LIBRES.items()]
    if "google" in synth.get("moteur", ""):
        sources.append({"nom": "Google Cloud Text-to-Speech fr-CA (OPTION activée)", "role": "voix de synthèse",
                        "licence": "conditions de Google Cloud, non tranchées pour un modèle distribué"})
    resume_variantes = [{k: v for k, v in x.items() if k != "tflite"} for x in faites]
    rapport = rapport_public_md(mot, fiche, version, synth, comptes, seuils, neuf, heures_gpu, sources, resume_variantes)
    (sortie / "rapport.md").write_text(rapport, encoding="utf-8")
    ecrire_json(sortie / "rapport.json", {"version": version, "seuils": seuils, "nouveau": neuf,
                                          "mesures_brutes": m_neuf, "heures_gpu": heures_gpu, "sources": sources,
                                          "comptes": comptes, "fiche": fiche, "variantes": resume_variantes})
    (sortie / "voice-pe.yaml").write_text(yaml_voice_pe(mot, seuils, f"/config/models/{mot}.json"), encoding="utf-8")
    ecrire_json(DONNEES / "mots" / mot / "dernier.json", {"version": version, "quand": dt.datetime.now().isoformat(),
                                                         "seuil": seuil_du_cran(seuils["principal"]),
                                                         "cran": seuils["principal"], "fenetre": fenetre,
                                                         "fa_h": neuf["fa_h"]})
    log.info("livré : %s\n%s", sortie, rapport)
    return sortie


def mots_demandes():
    if os.environ.get("EVEIL_MOT"):
        return [os.environ["EVEIL_MOT"]]
    return sorted(p.parent.name for p in DONNEES.glob("mots/*/fiche.json"))


def main():
    ap = argparse.ArgumentParser(description="Entraîne un mot d'éveil microWakeWord avec des données libres seulement.")
    ap.add_argument("--essai", action="store_true", help="petit et rapide : 4 voix, 300 pas, préchargement léger")
    ap.add_argument("--pas", type=int, help="nombre de pas d'entraînement")
    ap.add_argument("--sans-gpu", action="store_true", help="processeur seulement (très lent)")
    ap.add_argument("--precharger", action="store_true", help="seulement le préchargement")
    args = ap.parse_args()
    random.seed(0)
    garde_public()
    if args.precharger:
        precharger_public(leger=args.essai)
        return
    liste = mots_demandes()
    if not liste:
        log.info("aucun mot à entraîner (EVEIL_MOT, ou mots/<mot>/fiche.json dans %s)", DONNEES)
        return
    echecs = []
    for mot in liste:
        try:
            entrainer_mot(mot, essai=args.essai, pas=args.pas, sans_gpu=args.sans_gpu)
        except Exception as e:
            log.exception("échec pour %s", mot)
            echecs.append(f"{mot}: {e}")
            ecrire_json(DONNEES / "mots" / mot / "dernier.json", {"echec": str(e), "quand": dt.datetime.now().isoformat()})
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main()
