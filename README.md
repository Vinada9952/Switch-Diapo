# Switch Diapo

Un petit logiciel de contrôle de diaporama par mouvements de la main.

## Description

Ce projet détecte les mains via la webcam et permet de changer de diapositive avec des gestes:
- `poing` + balayage vers la droite → diapositive suivante
- `poing` + balayage vers la gauche → diapositive précédente

## Prérequis

- Python 3.10 ou supérieur
- Webcam
- Modules Python:
  - `opencv-python`
  - `mediapipe`
  - `pyautogui`

Installez-les avec:

```bash
pip install opencv-python mediapipe pyautogui
```

## Utilisation

1. Ouvrir un terminal dans le dossier du projet.
2. Lancer le script:

```bash
python switch-diapo.py
```

3. Choisir l'index de la caméra lorsque le programme le demande (par défaut `0`).
4. Fermer la fenêtre ou appuyer sur `q` pour quitter.

## Gestes

- `fermer la main` puis déplacer vers la droite puis `ouvrir la main` → diapositive suivante
- `fermer la main` puis déplacer vers la gauche puis `ouvrir la main` → diapositive précédente

## Exécutable

Un exécutable peut être généré avec PyInstaller via le script `build-exe.py`:

```bash
python build-exe.py
```

Assurez-vous d’avoir installé `pyinstaller` si vous souhaitez créer un exécutable.

## Remarques

- Le modèle MediaPipe `hand_landmarker.task` est téléchargé automatiquement si nécessaire.
- Si la détection est instable, rapprochez votre main de la caméra ou changez l’éclairage.
