import tensorflow as tf
import keras.src.saving.saving_lib
from dicewars import player
from random import choice

import random
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class Player(player.Player):
    """
    Modify the get_attack_areas function using your own player.

    An example of a player which plays random moves is implemented here
    """

    def __init__(self, state_size=None, action_size=None, MEMORY_SIZE=1, EPSILON=0, LEARNING_RATE=1e-9, BATCH_SIZE=32,
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
        self.target = self.build_model()

        if model == None:
            self.model = self.build_model()
        else:
            self.model = keras.src.saving.saving_lib.load_model(model)

    def build_model(self):
        """
        Create a simple neural network for DQN.
        """
        model = keras.Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def remember(self, grid, state, action, reward, next_state, done, valid_actions_next):
        """
         Store experience in memory.
          """
        from_player = state.player
        self.memory.append((self.better_state(grid, state, from_player), action, reward,
                            self.better_state(grid, next_state, from_player), done, valid_actions_next))


    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def replay(self):
        """ Train the model using replay memory. """
        if len(self.memory) < self.batch_size:
            return

        losses = []

        minibatch = random.sample(self.memory, self.batch_size)

        # Unpack batch elements
        states = np.array([x[0] for x in minibatch])
        actions = np.array([self.action_to_idx(x[1]) for x in minibatch])
        rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch], dtype=np.float32)
        dones = dones != -1 #this should also include whether player has been eliminated?
        valid_actions_next = [x[5] for x in minibatch]  # list of lists

        # Predict future rewards from target network
        future_rewards = self.target.predict(next_states, verbose=0)

        # Apply action masking
        masked_future_rewards = np.full_like(future_rewards, -np.inf)
        for i in range(self.batch_size):
            valid_idxs = self.actions_to_idxs(valid_actions_next[i])
            masked_future_rewards[i, valid_idxs] = future_rewards[i, valid_idxs]

        max_next_qs = np.max(masked_future_rewards, axis=1)

        # Compute Q values: Q = reward + gamma * max(Q_next) and set to -1 if done
        updated_qs = rewards + self.gamma * max_next_qs * (1 - dones)

        # Predict current Q-values
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)

            # Get Q-values for actions taken using one-hot masking
            action_masks = tf.one_hot(actions, self.action_size)
            q_action = tf.reduce_sum(q_values * action_masks, axis=1)

            # Compute loss
            loss = tf.keras.losses.Huber()(updated_qs, q_action)

        # Backpropagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return tf.reduce_mean(loss).numpy()

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

    def reward_state_old(self, old_state, new_state):
        reward = -0.1
        # player = old_state.player
        # old_dice = old_state.player_num_dice[player]
        # new_dice = new_state.player_num_dice[player]
        # reward += new_dice - old_dice
        #
        # if new_state.winner == player and player != -1:
        #     reward += 50
        #     print("Victory!")
        return reward

    def reward_state(self, old_state, new_state, scale = 1):
        player = old_state.player  # bc after you take an action it's not your turn anymore
        old_dice = old_state.player_num_dice[player]
        new_dice = new_state.player_num_dice[player]
        new_num_adjacent = new_state.player_max_size[player]
        old_num_adjacent = old_state.player_max_size[player]
        new_player_areas = new_state.player_areas
        old_player_areas = old_state.player_areas


        diminishing_factor = 1   # for now don't decrease with time
        reward = 0

        # Check if the new field increases the player's adjacent 
        if new_num_adjacent > old_num_adjacent:  
            reward += 0.01 * (new_num_adjacent - old_num_adjacent) * diminishing_factor  # Extra reward for forming larger groups

        # Check if you've eliminated an opponent (# of fields was not 0 and now is 0)
        for i in range(len(new_state.player_num_dice)):
            if i != player: 
                if len(new_player_areas[i]) == 0 and len(old_player_areas[i]) != 0:
                    reward += 0.2 * scale

        # Check for the average number of dice per area
        reward += (new_dice / len(new_player_areas[player])) * 0.02


        if new_dice > 0:
            reward += 0.01

        
        if new_state.winner == player and player != -1:
            reward += 1

        return reward
    