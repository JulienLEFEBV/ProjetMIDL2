# Rapport projet MIDL 2

## Introduction

Dans ce projet, nous nous sommes intéressés à l’analyse et à la prédiction de trajectoires du
mouvement humain à partir de vidéos.  
Ce type de problématique est central en vision par ordinateur, notamment pour des applications
comme l’analyse du geste, le sport, la biomécanique ou encore la danse.  
Nous avons choisi de nous concentrer sur le mouvement humain dans un contexte artistique,
plus précisément la danse, car elle offre des trajectoire continues, structurées et riches à
analyser.  
Parmi les différents types de mouvements possibles, nous avons volontairement choisi des
danses de style classique ou contemporain fluide.  
L’objectif était d’éviter les mouvements trop explosifs ou acrobatiques, comme le breakdance,
qui peuvent poser problème pour l’estimation de pose et rendre l’analyse des trajectoires moins
stable.  
La danse classique présente au contraire des mouvements lents, contrôlés, avec des
trajectoires continues des bras, des jambes et du centre du corps, ce qui est particulièrement
adapté à l’analyse temporelle.  
Pour travailler sur des données cohérentes et exploitables, nous avons utilisé un jeu de
données existant appelé AIST++, qui est un dataset de recherche dédié à la danse.
Ce dataset contient des vidéos de danse filmées avec une caméra fixe, en plan large, ce qui
permet de voir l’ensemble du corps du danseur, un point essentiel pour l’estimation de pose
avec MMPose.  
Même si AIST++ ne contient pas de ballet classique académique au sens strict, il propose
certains styles de danse fluide, proches du classique et du néo-classique, qui restent
compatibles avec notre objectif d’analyse de trajectoires continues.  
Le jeu de données a été récupéré depuis le site officiel du projet AIST++.  
Après avoir accepté les conditions d’utilisation, nous avons utilisé le script de téléchargement
fourni par les auteurs du dataset, ce qui permet de récupérer automatiquement les vidéos.  
Nous avons ensuite sélectionné manuellement un sous-ensemble de vidéos correspondant à
des styles de danse fluide et contrôlée, et exclu les styles plus explosifs comme le hip-hop ou le
breakdance.  
Ce choix de jeu de données nous permet donc de travailler sur des mouvements réalistes,
lisibles et cohérents avec les hypothèses de MMPose, tout en restant dans un cadre artistique
et contrôlé, adapté à l’analyse et à la prédiction de trajectoires.

## Utilisation de MMPOSE

Pour la prédiction des points nous avons choisi de faire tourner MMPose en local en utilisant
Cuda pour permettre notamment d’utiliser les modèles en 3D. Pour l'installation de MMPose
nous avons utilisé Conda comme indiqué dans le tutoriel officiel de MMPose pour simplifier
l’installation.
Nous avons utilisé le modèle human3d (l’alias du modèle
vid_pl_motionbert_8xb32-120e_h36m) qui permet une extraction de 17 points clés en 3D
sur un corps humain complet. Ces 17 keypoints suivent le standard Human3.6M.
Ainsi nous avons pu récupérer nos points sur l’ensemble de notre dataset et grâce au choix
de celui-ci, MMPose a donné des résultats très satisfaisants. En effet, tous nos scores de
prédiction sont à 1 sur l’ensemble du dataset.
En analysant les résultats fournis par MMPose, nous nous sommes rendu compte que le
point zéro qui correspond au bassin sert d’origine pour les axes x et y de nos key points. De
plus, nous avons remarqué que certains keypoints notamment ceux qui sont sur une même
partie du corps, comme le bras avec le poignet, le coude et l'épaule ou la jambe avec le
pied, le genou et la hanche, ont des mouvements corrélés. Aussi nous avons remarqué que
certains keypoints étaient plus mobiles que d’autres comme par exemple la position des
mains varient beaucoup plus que la position du bassin notamment. On peut faire aussi la
même remarque pour la vitesse et l'accélération qui sont plus importantes aux extrémités du
corps (comme les mains ou les pieds) que pour les autres parties (comme le bassin par
exemple). Enfin nous avons aussi remarqué que les hauteurs de nos points ne subissent
pas de grande variations, les parties hautes du corps restent en haut et les parties basses
restent en bas.

## Sources

[Lien vers le site d'AIST++](google.github.io/aistplusplus_dataset/factsfigures.html)
