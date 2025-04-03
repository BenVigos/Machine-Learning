import numpy as np
import random
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import AgressivePlayer, RandomPlayer
from playergroupX import Player  # Import the agent

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 64
EPISODES = 1000
EPSILON = 1.0  # Exploration factor
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9


def action_to_idx(action):
    if action == None:
        return 0
    else:
        return action[0]*30+action[1]

def actions_to_idxs(valid_actions_next):
    idxs = []
    for i in range(len(valid_actions_next)):
        idxs.append(action_to_idx(valid_actions_next[i]))

    return idxs


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self.build_model()

    def build_model(self):
        """ Create a simple neural network for DQN. """
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done, valid_actions_next):
        """ Store experience in memory. """
        self.memory.append((state, action, reward, next_state, done, valid_actions_next))

    def replay(self):
        """ Train the model using replay memory. """
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done, valid_actions_next in minibatch:
            target = reward
            if not done:
                next_q_values = self.model.predict(np.array([next_state]), verbose=0)[0]
                masked_q_values = np.full(self.action_size, -np.inf)
                next_valid_action_idxs = actions_to_idxs(valid_actions_next)
                masked_q_values[next_valid_action_idxs] = next_q_values[next_valid_action_idxs]
                target += GAMMA * np.max(masked_q_values)

            target_f = self.model.predict(np.array([state]), verbose=0)[0]
            target_f[action_to_idx(action)] = target
            self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def act(self, state, valid_actions):
        """ Select an action using epsilon-greedy strategy with action masking. """
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)


        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        masked_q_values = np.full(self.action_size, -np.inf)
        action_indices = [valid_actions.index(a) for a in valid_actions]
        masked_q_values[action_indices] = q_values[action_indices]

        return valid_actions[np.argmax(masked_q_values)]


def simple_state(match_state, from_player):
    num_dice = match_state.player_num_dice
    my_dice = num_dice[from_player]
    others = np.delete(np.array(num_dice), from_player)
    others = np.sort(others)
    state = np.insert(others,0, my_dice)

    return state

def get_valid_actions(grid, match_state):
    """
    REWRITE THIS FUNCTION FOR YOUR OWN MACHINE LEARNING AGENT
    """
    from_player = match_state.player  # the index of the current player
    player_areas = match_state.player_areas  # the areas belonging to each player
    area_num_dice = match_state.area_num_dice  # the amount of dice on each area

    # add ending the turn to the list of possibilities
    possible_attacks = [None]

    # loop over all areas in posession of the current player
    for from_area in player_areas[from_player]:

        # check if the area has more than 1 dice
        if area_num_dice[from_area] > 1:

            # loops over all neigbors of the current area
            for to_area in grid.areas[from_area].neighbors:
                # check if the neigboring area is not your own
                if to_area not in player_areas[from_player]:
                    # append the area to the possible attack options
                    possible_attacks.append((from_area, to_area))
    return possible_attacks

def reward_state(old_state, new_state):
    return 0


# Training loop
game = Game(num_seats=4)
match = Match(game)

player_agent = Player()  # Our DQN player
players = [player_agent, AgressivePlayer(), RandomPlayer(), RandomPlayer()]
state_size = 4  #simplest state, number of dice per player
action_size = len(game.area_num_dice) ** 2  # Assuming all possible (from, to) moves

agent = DQNAgent(state_size, action_size)


for episode in range(EPISODES):
    match = Match(Game(num_seats=4))
    player = match.player
    state = simple_state(match.state, player)
    full_state = match.state
    done = False
    while not done:
        player = match.player
        valid_actions = get_valid_actions(match.game.grid, match.state)
        action = agent.act(state, valid_actions)
        grid, new_state = match.step(action)
        full_new_state = new_state
        print(full_state.player)
        reward = reward_state(full_state, full_new_state)
        done = new_state.winner != -1

        next_state = simple_state(new_state, player)
        valid_actions_next = get_valid_actions(match.game.grid, new_state)

        agent.remember(state, action, reward, next_state, done, valid_actions_next)
        state = next_state
        full_state = full_new_state

    agent.replay()

    print(f"Episode {episode + 1}/{EPISODES} - Epsilon: {agent.epsilon:.2f}")

agent.model.save("dqn_model.h5")
print("Training complete. Model saved!")
