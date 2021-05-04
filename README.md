# TP5_ObjectConnecte
TP5 on reinforcement learning
Explication de l'apprentissage par renforcement avec Open AI gym

Environnement choisi: Cartpole-v1

Bref description de l'environnement

- Condtion de réussite: Réussir à garder le baton stable pendant 500 frames (Cartpole-v1) ou 200 (cartpole-v0)
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
    - qTable : Tableaux contenant les probilité pour la meilleur action possibles pour chaque états

Functionnement:
    1 - instanciation de la q-table avec des valeurs aléatoires entre 0 et -2
    2 - réinitialisation de l'environnemnt, nous retourne un état aléatoire
    3 - Selon la valeur du epsilon va faire une action aléatoire ou une action de la q-table
    4 - Va chercher l'état future
    5 - Va chercher les valeurs de la qtable pour l'état présent et l'état future
    6 - Calcule une nouvelle probabilté pour l'action faite et la remplace dans la q-table
    7 - Si la condition d'échec n'a pas été activé



    ----------------------------------------------    
    yo
