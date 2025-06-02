import sys
import gymnasium as gym
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Funzione per preprocessare le osservazioni
def preprocess_observation(obs):
    # Ridimensionamento dell'immagine a 96x96 pixel
    obs_resized = cv2.resize(obs, (96, 96))
    obs_normalized = obs_resized.astype('float32') / 255.0
    return obs_normalized

# Funzione per mappare l'azione discreta a un'azione continua
def discrete_to_continuous(action):
    # La mappa delle azioni per un ambiente continuo
    action_map = {
        0: np.array([0.0, 0.0, 0.0]),   # Do nothing
        1: np.array([0.3, 0.0, 0.0]),   # Steer left
        2: np.array([-0.3, 0.0, 0.0]),  # Steer right
        3: np.array([0, 0.5, 0]),       # Accelerate
        4: np.array([0.0, 0.0, 0.8])    # Brake
    }
    return action_map[action]

# Funzione per simulare l'ambiente
def play(env, model):
    """
    Simula l'ambiente CarRacing-v3 utilizzando un modello addestrato.
    """
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # Scarta i primi 50 frame
    for _ in range(50):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))  # Azione continua "do nothing"

    done = False
    while not done:
        # Preprocessing dell'osservazione
        obs_preprocessed = preprocess_observation(obs)
        obs_batch = np.expand_dims(obs_preprocessed, axis=0)  # Aggiunge la dimensione batch
        
        # Predizione dell'azione discreta (l'output del modello è un vettore di probabilità)
        action_probabilities = model.predict(obs_batch)
        discrete_action = np.argmax(action_probabilities)  # Azione discreta con probabilità massima
        
        # Converti l'azione discreta in un'azione continua
        action = discrete_to_continuous(discrete_action)
        
        # Esegui l'azione nell'ambiente
        obs, reward, terminated, truncated, _ = env.step(action)  # Azione continua
        done = terminated or truncated

# Configurazione dell'ambiente
env_arguments = {
    'domain_randomize': False,
    'continuous': True,  # Azioni continue
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
