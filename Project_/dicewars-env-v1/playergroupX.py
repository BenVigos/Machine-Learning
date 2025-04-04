from dicewars import player
from random import choice
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


class Player(player.Player):
    """
    Modify the get_attack_areas function using your own player.

    An example of a player which plays random moves is implemented here
    """

    def __init__(self, state_size=None, action_size=None, MEMORY_SIZE=1, EPSILON=0, LEARNING_RATE=0, BATCH_SIZE=32,
                 GAMMA=0, EPSILON_MIN=0, EPSILON_DECAY=0, model=None):
        """
        do all required initialization here 
        use relative paths for access to stored files that you require
        use self.variable to store your variables such that your class has access.
        """
        self.playername = 'Group X'
        print(f'Initializing player from: {__file__} with name:', self.playername)

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        if model == None:
            self.model = self.build_model()
        else:
            self.epsilon = 0
            self.model = None  # Change to load model

    def build_model(self):
        """ Create a simple neural network for DQN. """
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, valid_actions_next):
        """ Store experience in memory. """
        from_player = state.player
        self.memory.append((self.simple_state(state, from_player), action, reward,
                            self.simple_state(next_state, from_player), done, valid_actions_next))

    def replay(self):
        """ Train the model using replay memory. """
        if len(self.memory) < self.batch_size:
            return

        total = 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done, valid_actions_next in minibatch:
            target = reward
            total+=reward
            if not done:
                next_q_values = self.model.predict(np.array([next_state]), verbose=0)[0]
                masked_q_values = np.full(self.action_size, -np.inf)
                next_valid_action_idxs = self.actions_to_idxs(valid_actions_next)
                masked_q_values[next_valid_action_idxs] = next_q_values[next_valid_action_idxs]
                target += self.gamma * np.max(masked_q_values)

            target_f = self.model.predict(np.array([state]), verbose=0)[0]
            target_f[self.action_to_idx(action)] = target
            self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"Avg reward: {total/len(self.memory)}")

    def get_valid_actions(self, grid, match_state):
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

    def get_attack_areas(self, grid, match_state):
        """ Select an action using epsilon-greedy strategy with action masking. """
        valid_actions = self.get_valid_actions(grid, match_state)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        match_state = self.simple_state(match_state, match_state.player)

        q_values = self.model.predict(np.array([match_state]), verbose=0)[0]
        masked_q_values = np.full(self.action_size, -np.inf)
        action_indices = [valid_actions.index(a) for a in valid_actions]
        masked_q_values[action_indices] = q_values[action_indices]

        return valid_actions[np.argmax(masked_q_values)]

    def simple_state(self, match_state, from_player):
        num_dice = match_state.player_num_dice
        my_dice = num_dice[from_player]
        others = np.delete(np.array(num_dice), from_player)
        others = np.sort(others)
        state = np.insert(others, 0, my_dice)

        return state

    def action_to_idx(self, action):
        if action == None:
            return 0
        else:
            return action[0] * 30 + action[1]

    def actions_to_idxs(self, valid_actions_next):
        idxs = []
        for i in range(len(valid_actions_next)):
            idxs.append(self.action_to_idx(valid_actions_next[i]))

        return idxs

    def reward_state(self, old_state, new_state):
        player = old_state.player
        old_dice = old_state.player_num_dice[player]
        new_dice = new_state.player_num_dice[player]
        return new_dice - old_dice
