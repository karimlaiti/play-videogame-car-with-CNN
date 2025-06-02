import sys
import gymnasium as gym
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gymnasium.wrappers import RecordVideo
import pyautogui
import time
import os



# Funzione per preprocessare le osservazioni
def preprocess_observation(obs):

    obs_resized = cv2.resize(obs, (96,96))
    obs_normalized = obs_resized.astype('float32') / 255.0
    return obs_normalized

# Funzione per convertire azioni discrete in azioni continue
def discrete_to_continuous(action):
 
    action_map = {
        0: np.array([0.0, 0.0, 0.0]),  # Do nothing
        1: np.array([0.5, 0.0, 0.2]), # Steer left
        2: np.array([-0.5, 0.0, 0.2]),  # Steer right
        3: np.array([0, 0.2, 0]),  # Accelerate
        4: np.array([0.0, 0.0, 0.2])   # Brake
    }
    return action_map[action]

# Funzione per simulare l'ambiente
def play(env, model):
    """
    Simula l'ambiente CarRacing-v3 utilizzando un modello addestrato.
    """
    seed = 2500
    obs, _ = env.reset(seed=seed)
    
    # Scarta i primi 50 frame
    for _ in range(50):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))  # Azione continua "do nothing"
    
    done = False
    while not done:
        # Preprocessing dell'osservazione
        obs_preprocessed = preprocess_observation(obs)
        obs_batch = np.expand_dims(obs_preprocessed, axis=0)  # Aggiunge la dimensione batch
        
        # Predizione dell'azione
        probabilities = model.predict(obs_batch)
        discrete_action = np.argmax(probabilities)  # Azione discreta con probabilità massima
        
        # Converti l'azione discreta in un'azione continua
        continuous_action = discrete_to_continuous(discrete_action)
        
        # Esegui l'azione nell'ambiente
        obs, _, terminated, truncated, _ = env.step(continuous_action)
        done = terminated or truncated

# Configurazione dell'ambiente
env_arguments = {
    'domain_randomize': False,
    'continuous': True,  # L'ambiente accetta solo azioni continue
    'render_mode': 'human'  # Mostra la simulazione
}

env_name = 'CarRacing-v3'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Percorso al modello addestrato
model_path = r"C:\Users\karim\OneDrive - uniroma1.it\Documents\machine learning\homework2\my_model.h5"

# Carica il modello
model = load_model(model_path)
print("Modello caricato con successo.")

# Avvia la simulazione
play(env, model)

