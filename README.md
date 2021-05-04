# TP5_ObjectConnecte
TP5 on reinforcement learning
Explication de l'apprentissage par renforcement avec Open AI gym

Environnement choisi: Cartpole-v1

Bref description de l'environnement

- Condtion de réussite: Réussir à garder un baton à la vertical stable pendant 500 frames (Cartpole-v1) ou 200 (cartpole-v0)
- Conditions d'échec  si le baton à un angle ±12° | si la possition du cart dépasse ±2.4
- Observation Tableau[4]: Cart position, Cart Velocity, Pole angle, Pole velocity at tip
- Actions possibles: droite = 1, gauche = 0

Algorithme d'apprentissage par renforcement: Q-Learning 

Bref description des variables importantes:
    - LEARNING_RATE / LEARNING_RATE_DECAY: Comment la nouvelle info va écraser la vielle info, valeur entre 0 et 1. . La valeur décroît au cours de l'apprentissage
    - epsilon / epsilon_decay_value : Mesure qui indique la chance que l'agent choisi une action aléatoire au lieu de d'exploiter ses connaissances. La valeur décroît au cours de l'apprentissage
    - START_DECAYING / STOP_DECAYING : Début et fin de la période de décroissance 
    - DISCOUNT :  multiplicateur qui indique l'importance de la valeur future entre 0 et 1
    - numBins : Nombre de sections pour les observations
    - qTable : Tableaux contenant les probilités pour la meilleur action possibles pour chaques états

Functionnement:
     1 - Instanciation de la q-table avec des valeurs aléatoires entre 0 et -2
     2 - Réinitialisation de l'environnement, nous retourne un état aléatoire
     3 - Selon la valeur du epsilon va faire une action aléatoire ou une action de la q-table
     4 - Va chercher l'état future
     5 - Va chercher les valeurs de la qtable pour l'état présent et l'état future
     6 - Calcule une nouvelle probabilté pour l'action et la remplace dans la q-table
     7 - Si la condition d'échec n'a pas été activé assigne le nouveau état en temps que état présent
     9 - Recomense temps que l'épisode n'est pas fini
     8 - Quand l'épisode fini et que les condition de StART_DECAY et END_DECAY sont respectée soustrait le le DECAY_RATE au LEARNING_RATE et l'epsilon
    10 - Répete selon le nombre épisodes 



