import keras.src.saving.saving_lib
from dicewars import player
from random import choice
import random
import numpy as np
from tensorflow.keras.models import Sequential, load_model
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
            self.model = keras.src.saving.saving_lib.load_model(model)

    def build_model(self):
        """
        Create a simple neural network for DQN.
        """
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, grid, state, action, reward, next_state, done, valid_actions_next):
        """
         Store experience in memory.
          """
        from_player = state.player
        self.memory.append((self.better_state(grid, state, from_player), action, reward,
                            self.better_state(grid, next_state, from_player), done, valid_actions_next))

    def replay(self):
        """ Train the model using replay memory. """
        if len(self.memory) < self.batch_size:
            return

        losses = []

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done, valid_actions_next in minibatch:
            target = reward
            if not done:
                next_q_values = self.model.predict(np.array([next_state]), verbose=0)[0]
                masked_q_values = np.full(self.action_size, -np.inf)
                next_valid_action_idxs = self.actions_to_idxs(valid_actions_next)
                masked_q_values[next_valid_action_idxs] = next_q_values[next_valid_action_idxs]
                target += self.gamma * np.max(masked_q_values)

            target_f = self.model.predict(np.array([state]), verbose=0)[0]
            action_idx = self.action_to_idx(action)
            target_old = target_f[action_idx]
            target_f[action_idx] = target

            # Compute loss manually (MSE between target and prediction for selected action)
            loss = (target - target_old) ** 2
            losses.append(loss)

            self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        avg_loss = np.mean(losses)
        print(f"Replay: Avg loss = {avg_loss:.4f}")

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

        match_state = self.better_state(grid, match_state, match_state.player)

        q_values = self.model.predict(np.array([match_state]), verbose=0)[0]
        masked_q_values = np.full(self.action_size, -np.inf)
        action_indices = [valid_actions.index(a) for a in valid_actions]
        masked_q_values[action_indices] = q_values[action_indices]

        return valid_actions[np.argmax(masked_q_values)]

    def simple_state(self, match_state, from_player):
        """
        Simple state representation. Only shows the total # of dice per player.
        "My" dice always go first and the rest are ranked from least to most
        """
        num_dice = match_state.player_num_dice
        my_dice = num_dice[from_player]
        others = np.delete(np.array(num_dice), from_player)
        others = np.sort(others)
        state = np.insert(others, 0, my_dice)

        return state

    def better_state(self, grid, match_state, from_player, max_neighbors=5):
        import numpy as np

        num_areas = len(match_state.area_players)
        area_dice = match_state.area_num_dice
        area_owners = match_state.area_players

        # Step 1: Build raw feature list before sorting
        raw_features = []

        for area_idx in range(num_areas):
            owner_flag = 1 if area_owners[area_idx] == from_player else -1
            neighbors = grid.areas[area_idx].neighbors
            sorted_neighbors = sorted(neighbors, key=lambda x: -area_dice[x])

            # Pad neighbors with -1
            padded_neighbors = sorted_neighbors[:max_neighbors]
            while len(padded_neighbors) < max_neighbors:
                padded_neighbors.append(-1)

            raw_features.append({
                "original_idx": area_idx,
                "dice": area_dice[area_idx],
                "owner": owner_flag,
                "neighbors": padded_neighbors
            })

        # Step 2: Sort features and create new index mapping
        sorted_features = sorted(raw_features, key=lambda x: (-x["owner"], -x["dice"]))
        index_map = {feat["original_idx"]: i for i, feat in enumerate(sorted_features)}

        # Step 3: Remap neighbors to new sorted indices
        area_vector_list = []
        for feat in sorted_features:
            remapped_neighbors = [
                index_map[n] if n in index_map else -1 for n in feat["neighbors"]
            ]
            area_vector_list.append([
                feat["dice"],
                feat["owner"],
                *remapped_neighbors
            ])

        # Step 4: Flatten to 1D array
        return np.array(area_vector_list, dtype=np.float32).flatten()

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
        reward = 1
        player = old_state.player
        old_dice = old_state.player_num_dice[player]
        new_dice = new_state.player_num_dice[player]
        if new_dice - old_dice > 0 :
            reward += new_dice - old_dice

        if new_state.winner == player and player != -1:
            reward += 1000
            print("Victory!")
        return reward
