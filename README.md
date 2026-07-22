# Switch Diapo

Un petit logiciel de contrôle de diaporama par mouvements de la main.

## Description

Ce projet sépare la détection de la main et le contrôle de diapositive :
- `hand_detection.py` détecte les gestes avec la webcam via MediaPipe.
- `diapo_switcher.py` écoute des requêtes HTTP et simule des appuis sur les touches `espace` et `backspace` avec `pyautogui`.

Le flux de base est le suivant :
1. Fermer la main (`poing`).
2. Balayer vers la droite ou vers la gauche.
3. Ouvrir la main pour valider le changement de diapositive.

## Prérequis

- Python 3.10 ou supérieur
- Webcam
- Modules Python :
  - `opencv-python`
  - `mediapipe`
  - `pyautogui`
  - `requests`
  - `flask`

Installez-les avec :

```bash
pip install opencv-python mediapipe pyautogui requests flask
```

Si vous souhaitez créer un exécutable, installez également `pyinstaller` :

```bash
pip install pyinstaller
```

## Utilisation

1. Ouvrir un terminal dans le dossier du projet.
2. Lancer le serveur de contrôle de diaporama :

```bash
python diapo_switcher.py
```

3. Dans un autre terminal, lancer la détection de main :

```bash
python hand_detection.py
```

4. Choisir l'index de la caméra lorsque le programme le demande (par défaut `0`).
5. Appuyer sur `q` dans la fenêtre de prévisualisation pour quitter.

## Configuration de l'adresse IP

Le script `hand_detection.py` demande l'adresse du serveur HTTP :
- Valeur par défaut : `http://127.0.0.1:9952`
- Si vous utilisez un autre hôte ou port, entrez l'URL complète.

## Gestes supportés

- `poing` + balayage vers la droite + ouverture de la main → diapositive suivante
- `poing` + balayage vers la gauche + ouverture de la main → diapositive précédente

## Génération d'un exécutable

Un exécutable peut être créé avec PyInstaller via le script `build-exe.py` :

```bash
python build-exe.py
```

Ce script ajoute automatiquement un hook temporaire pour MediaPipe et génère un fichier unique avec PyInstaller.

## Notes importantes

- Le modèle MediaPipe `hand_landmarker.task` est téléchargé automatiquement dans un dossier utilisateur local.
- Évitez les chemins contenant des accents ou des espaces si vous rencontrez des problèmes avec MediaPipe sous Windows.
- Si la détection est instable, rapprochez votre main de la caméra ou améliorez l’éclairage.

## Structure des fichiers

- `hand_detection.py` : détection de gestes et envoi des requêtes HTTP au serveur.
- `diapo_switcher.py` : serveur Flask qui simule les touches du clavier.
- `build-exe.py` : script pour générer un exécutable PyInstaller.
- `hand_landmarker.task` : modèle MediaPipe (téléchargé automatiquement).
