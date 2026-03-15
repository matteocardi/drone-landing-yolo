# drone-landing-yolo

Projet Deep Learning

En résumé, j'ai entraîné un détecteur YOLO à reconnaître un pad d'atterrissage orange
à partir de 182 photos prises à la main. Le modèle tourne ensuite en temps réel sur webcam et décide si la scène est safe pour atterrir, selon la confiance de détection et la taille apparente du pad dans l'image.

Contenu du repo :

drone-landing-yolo/
├── README.md
├── Projet_train.ipynb    dataset → entraînement → évaluation → seuils
├── demo_live.py          Script webcam temps réel
└── logs/
    └── demo_20260314_202756.csv     Log des scénarios enregistrés pendant la démo


Sur le drive au liens : https://drive.google.com/drive/folders/1ffTeQiwvH3ZeonsOP5iTK2mqTC2eYhAT?usp=sharing
On retrouve :
-DataSet complet (images + labels + data.yaml)
-Fichier drone_landing pour lancer camera live (best.pt + demo_live.py)
-Vidéo présentation orale
-Vidéo démonstration live
-Fichier d'entrainement (Projet_train)

Pour reproduire l'entrainement :

- Ouvrir `Projet_train.ipynb` sur collab et activer le GPU

- Modifier ces lignes avec votre DataSet :
rf = Roboflow(api_key="Numero_clef")
project = rf.workspace("matteos-workspace-f2qmj").project("landing_pad-nm9u3")
version = project.version(2)
dataset = version.download("yolov11")
(ces clefs sont celles de mon dataset)

-Lancer l'entrainement avec vos parametres
model = YOLO("yolo11n.pt")
model.train(
    data="landing_pad-2/data.yaml",
    imgsz=640,
    epochs=50,
    batch=16,
    patience=20
)

-Evaluer le test avec les metrics

-Lancer la demo en téléchargeant dans le terminal (pip install ultralytics opencv-python) et télécharger `best.pt` depuis le Google Drive et le placer dans le même dossier que `demo_live.py`. 
dans le terminal de ce dossier, lancer "python demo_live.py"
La caméra se lance, appuyer sur "S" pour logger le scénario actuel dans le CSV et "Q" pour quitter.


Résultats obtenus :
Test set (21 images) :
- Précision   : 0.996
- Rappel      : 1.000
- mAP@0.5     : 0.995
- mAP@0.5:0.95: 0.829

Seuils de décision (gelés sur val set) :
- τ_conf = 0.50  (confiance minimale du détecteur)
- τ_area = 0.01  (pad doit occuper au moins 1% de l'image)
Décision finale test set : 21/21 SAFE


Note sur IA :
Nous avons utilisé Claude pour le débogage, l'interprétation des métriques, la structuration du pipeline.
Chaque étape a été comprise et validée manuellement avant d'être exécutée.
