"""Les garde-fous du mode public (python3 -I -m unittest entraineur/test_public.py).

Ce que ces tests tiennent, pour qu'un changement futur ne le défasse pas sans
que la CI rougisse :
  - l'entraîneur REFUSE un dossier de données qui contient du privé ;
  - la garde passe avant tout geste (précharger, synthèse, entraîner) ;
  - le code n'a aucun chemin qui lise un dossier privé ;
  - aucune source distribuée n'est NC, YouTube ou « voir le dépôt ».
Il ne faut que numpy (importé par entrainer.py) : pas de TensorFlow, pas de GPU.
"""

import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path

ICI = Path(__file__).resolve().parent


def charger():
    spec = importlib.util.spec_from_file_location("entrainer_sous_test", ICI / "entrainer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Garde(unittest.TestCase):
    def setUp(self):
        self.e = charger()
        self.tmp = tempfile.TemporaryDirectory()
        self.racine = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_dossier_propre_passe(self):
        (self.racine / "mots" / "essai").mkdir(parents=True)
        (self.racine / "modeles").mkdir()
        self.assertEqual(self.e.garde_public(self.racine), self.racine)

    def test_chaque_dossier_prive_est_refuse(self):
        for nom in self.e.DOSSIERS_PRIVES:
            with self.subTest(nom=nom):
                d = self.racine / nom
                d.mkdir()
                with self.assertRaises(self.e.ErreurPublique) as r:
                    self.e.garde_public(self.racine)
                self.assertIn(nom, str(r.exception))
                d.rmdir()

    def test_les_quatre_dossiers_prives_connus(self):
        self.assertEqual(set(self.e.DOSSIERS_PRIVES), {"prises", "ambiance", "reveils", "repas"})

    def test_entrainer_mot_refuse_avant_tout_geste(self):
        e = self.e
        (self.racine / "prises" / "un_mot" / "locuteur-1" / "positif").mkdir(parents=True)
        e.DONNEES = self.racine

        def piege(*a, **k):
            raise AssertionError("un geste a eu lieu avant la garde")

        e.precharger_public = e.synthese_publique = e.synthese_piper = piege
        with self.assertRaises(e.ErreurPublique):
            e.entrainer_mot("un_mot", essai=True, sans_gpu=True)

    def test_main_refuse_avant_tout_geste(self):
        e = self.e
        (self.racine / "ambiance").mkdir()
        e.DONNEES = self.racine
        import sys
        ancien = sys.argv
        sys.argv = ["entrainer.py", "--precharger"]
        try:
            e.precharger_public = lambda *a, **k: self.fail("préchargement avant la garde")
            with self.assertRaises(e.ErreurPublique):
                e.main()
        finally:
            sys.argv = ancien

    def test_le_code_ne_lit_aucun_dossier_prive(self):
        # Aucun glob ni chemin vers prises/, ambiance/ ou reveils/ hors de la
        # liste de la garde elle-même.
        source = (ICI / "entrainer.py").read_text(encoding="utf-8")
        sans_garde = re.sub(r"DOSSIERS_PRIVES = \(.*?\)\n", "", source)
        for nom in ("prises", "ambiance", "reveils", "repas"):
            for motif in (f'"{nom}/', f"'{nom}/", f'/ "{nom}"', f"/ '{nom}'", f'DONNEES / "{nom}'):
                self.assertNotIn(motif, sans_garde, f"chemin privé {nom} dans le code")


class Sources(unittest.TestCase):
    def setUp(self):
        self.e = charger()

    def test_sources_libres_redistribuables(self):
        e = self.e
        for nom, (_, licence, _) in e.SOURCES_LIBRES.items():
            with self.subTest(nom=nom):
                self.assertNotIn(nom, e.SOURCES_REFUSEES_EN_PUBLIC)
                self.assertNotRegex(licence, r"\bNC\b|-nc-|YouTube|voir le d", "licence non redistribuable")
                self.assertRegex(licence, r"CC BY|CC0|MIT|Apache|CONFIRMER")
        for nom in e.SOURCES_REFUSEES_EN_PUBLIC:
            self.assertNotIn(nom, e.SOURCES_LIBRES)

    def test_precharger_ne_touche_pas_aux_sources_refusees(self):
        import inspect
        corps = inspect.getsource(self.e.precharger_public) + "".join(
            inspect.getsource(getattr(self.e, f)) for f in ("_precharger_musan", "_precharger_demand", "_precharger_cv",
                                                            "_precharger_fleurs", "_precharger_parole_plus"))
        for mot in ("fma", "audioset", "negatifs", "NEGATIFS_OFFICIELS", "dinner_party"):
            self.assertNotIn(mot, corps.replace("MUSAN", "").replace("music", ""), mot)

    def test_phrases_generiques_sans_le_mot(self):
        e = self.e
        g = e.phrases_generiques({"phrase": "Hé Cumulus"})
        self.assertNotIn("Hé Cumulus", g)
        self.assertIn("Cumulus", g)  # chaque mot seul est un négatif
        g2 = e.phrases_generiques({"phrase": "Ok Google"})
        self.assertNotIn("Ok Google", g2)

    def test_google_reste_une_option(self):
        e = self.e
        e.GOOGLE_CLE = "cle-de-test"
        os.environ.pop("EVEIL_PUBLIC_GOOGLE", None)
        appels = []
        e.synthese_piper = lambda f, essai=False: {"positif": [], "negatif": [], "generiques": [], "voix": 0, "caracteres": 0, "moteur": "piper"}
        self.assertEqual(e.synthese_publique({"phrase": "x"})["moteur"], "piper")  # clé présente mais pas demandée
        os.environ["EVEIL_PUBLIC_GOOGLE"] = "1"
        try:
            e.synthese_google = lambda f, essai=False: appels.append(1) or {"positif": [], "negatif": [], "voix": 1, "caracteres": 1}
            self.assertEqual(e.synthese_publique({"phrase": "x"})["moteur"], "piper+google")
            self.assertEqual(appels, [1, 1])  # positifs, puis négatifs génériques
        finally:
            os.environ.pop("EVEIL_PUBLIC_GOOGLE", None)


if __name__ == "__main__":
    unittest.main()
