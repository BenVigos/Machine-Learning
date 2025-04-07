import tensorflow as tf
import keras.src.saving.saving_lib
from dicewars import player
import random
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from collections import deque


class Player(player.Player):
    def __init__(self, state_size = 22, action_size=10, MEMORY_SIZE=10000, EPSILON=1.0, LEARNING_RATE=1e-3,
                 BATCH_SIZE=64, GAMMA=0.99, EPSILON_MIN=0.01, EPSILON_DECAY=0.995, model=None):

        self.playername = 'Group X - Local'
        print(f'Initializing player with local representation: {__file__}')

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

        if model is None:
            self.model = self.build_model()
        else:
            self.model = keras.src.saving.saving_lib.load_model(model)

    def build_model(self):
        model = Sequential([
            Dense(64, activation="relu", input_shape=(self.state_size,)),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def get_local_features(self, grid, match_state, from_player, area_idx, max_neighbors=10):
        dice = match_state.area_num_dice[area_idx]
        owner_flag = 1 if match_state.area_players[area_idx] == from_player else -1
        neighbors = grid.areas[area_idx].neighbors
        neighbor_data = []

        valid_targets = []
        for n in sorted(neighbors, key=lambda x: -match_state.area_num_dice[x])[:max_neighbors]:
            n_owner = 1 if match_state.area_players[n] == from_player else -1
            n_dice = match_state.area_num_dice[n]
            neighbor_data.extend([n_owner, n_dice])
            valid_targets.append(n if n_owner == -1 else None)  # only enemies are valid targets

        while len(neighbor_data) < max_neighbors * 2:
            neighbor_data.extend([0, 0])
            valid_targets.append(None)

        return np.array([dice, owner_flag] + neighbor_data, dtype=np.float32), valid_targets

    def remember(self, grid, state, action, reward, next_state, done, valid_actions_next):
        self.memory.append((grid, state, action, reward, next_state, done, valid_actions_next))

    def update_target(self):
        self.target.set_weights(self.model.get_weights())


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        local_inputs = []
        action_indices = []
        rewards = []
        next_state_inputs = []
        mask_list = []
        map_current_to_future = []

        for (grid, state, action, reward, next_state, done, valid_actions_next) in minibatch:
            if action is None:
                continue

            from_player = state.player
            from_area, to_area = action

            local_input, valid_targets = self.get_local_features(grid, state, from_player, from_area)
            if to_area not in valid_targets:
                continue

            action_idx = valid_targets.index(to_area)

            # Save input and action
            local_inputs.append(local_input)
            action_indices.append(action_idx)
            rewards.append(reward)

            # Process next state (batch later)
            if not done:
                next_inputs_per_sample = []
                masks_per_sample = []
                for a in valid_actions_next:
                    if a is None:
                        continue
                    next_from, _ = a
                    next_input, valid_targets_next = self.get_local_features(grid, next_state, from_player, next_from)
                    mask = [i for i, tgt in enumerate(valid_targets_next) if tgt is not None]
                    if mask:
                        next_inputs_per_sample.append(next_input)
                        masks_per_sample.append(mask)

                if next_inputs_per_sample:
                    next_state_inputs.extend(next_inputs_per_sample)
                    mask_list.extend(masks_per_sample)
                    map_current_to_future.append(len(next_inputs_per_sample))
                else:
                    map_current_to_future.append(0)
            else:
                map_current_to_future.append(0)

        if not local_inputs:
            return

        local_inputs = np.array(local_inputs)
        action_indices = np.array(action_indices)
        rewards = np.array(rewards)

        # Batched prediction
        predicted_q_values = self.model.predict(local_inputs, verbose=0)

        if next_state_inputs:
            future_preds = self.target.predict(np.array(next_state_inputs), verbose=0)
        else:
            future_preds = []

        # Calculate target Q values
        targets = np.copy(predicted_q_values)
        pred_ptr = 0
        for i in range(len(local_inputs)):
            if map_current_to_future[i] > 0:
                sample_preds = future_preds[pred_ptr: pred_ptr + map_current_to_future[i]]
                sample_masks = mask_list[pred_ptr: pred_ptr + map_current_to_future[i]]
                masked_max_qs = [np.max(pred[m]) for pred, m in zip(sample_preds, sample_masks)]
                max_future_q = max(masked_max_qs)
                pred_ptr += map_current_to_future[i]
                targets[i, action_indices[i]] = rewards[i] + self.gamma * max_future_q
            else:
                targets[i, action_indices[i]] = rewards[i]

        # Convert to tensors for training
        local_inputs_tf = tf.convert_to_tensor(local_inputs, dtype=tf.float32)
        targets_tf = tf.convert_to_tensor(targets, dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = self.model(local_inputs_tf, training=True)
            loss = tf.keras.losses.Huber()(targets_tf, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.numpy()


    def replay_(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        local_inputs = []
        target_qs = []

        # To batch predict all next-state Q-values in one go
        all_next_inputs = []
        all_masks = []
        idx_mapping = []

        for idx, (grid, state, action, reward, next_state, done, valid_actions_next) in enumerate(minibatch):
            if action is None:
                continue

            from_player = state.player
            from_area, to_area = action

            local_input, valid_targets = self.get_local_features(grid, state, from_player, from_area)
            if to_area not in valid_targets:
                continue

            action_idx = valid_targets.index(to_area)

            next_inputs = []
            next_masks = []
            for a in valid_actions_next:
                if a is None:
                    continue
                next_from, _ = a
                ni, vt = self.get_local_features(grid, next_state, from_player, next_from)
                mask = [i for i, tgt in enumerate(vt) if tgt is not None]
                if mask:
                    next_inputs.append(ni)
                    next_masks.append(mask)

            if not done and next_inputs:
                all_next_inputs.extend(next_inputs)
                all_masks.extend(next_masks)
                idx_mapping.append((len(local_inputs), len(next_inputs)))  # map to where this sample's future preds will be
            else:
                idx_mapping.append((len(local_inputs), 0))

            local_inputs.append(local_input)
            target_qs.append((action_idx, reward))

        if not local_inputs:
            return

        future_preds = self.target.predict(np.array(all_next_inputs), verbose=0) if all_next_inputs else []

        targets = []
        pred_index = 0
        for i, (local_input, (action_idx, reward)) in enumerate(zip(local_inputs, target_qs)):
            target_q_vec = self.model.predict(np.array([local_input]), verbose=0)[0]
            _, n_inputs = idx_mapping[i]

            if n_inputs > 0:
                sample_future_preds = future_preds[pred_index:pred_index + n_inputs]
                sample_masks = all_masks[pred_index:pred_index + n_inputs]
                masked_max_qs = [np.max(p[m]) for p, m in zip(sample_future_preds, sample_masks)]
                max_future_q = max(masked_max_qs)
                target_q = reward + self.gamma * max_future_q
                pred_index += n_inputs
            else:
                target_q = reward

            target_q_vec[action_idx] = target_q
            targets.append(target_q_vec)

        local_inputs = np.array(local_inputs)
        target_qs = np.array(targets)

        with tf.GradientTape() as tape:
            predictions = self.model(local_inputs, training=True)
            loss = tf.keras.losses.Huber()(target_qs, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.numpy()

    def get_valid_actions(self, grid, match_state):
        from_player = match_state.player
        player_areas = match_state.player_areas[from_player]
        area_num_dice = match_state.area_num_dice
        possible_attacks = [None]

        for from_area in player_areas:
            if area_num_dice[from_area] > 1:
                for to_area in grid.areas[from_area].neighbors:
                    if to_area not in player_areas:
                        possible_attacks.append((from_area, to_area))
        return possible_attacks

    def get_attack_areas(self, grid, match_state):
        from_player = match_state.player
        player_areas = match_state.player_areas[from_player]
        area_num_dice = match_state.area_num_dice

        if np.random.rand() <= self.epsilon:
            return random.choice(self.get_valid_actions(grid, match_state))

        # Prepare batch input
        inputs = []
        area_refs = []

        for from_area in player_areas:
            if area_num_dice[from_area] <= 1:
                continue
            local_input, valid_targets = self.get_local_features(grid, match_state, from_player, from_area)
            inputs.append(local_input)
            area_refs.append((from_area, valid_targets))

        if not inputs:
            return None

        inputs = np.array(inputs)
        q_values_batch = self.model.predict(inputs, verbose=0)

        best_q = -np.inf
        best_action = None

        for i, (from_area, valid_targets) in enumerate(area_refs):
            q_values = q_values_batch[i]
            for j, to_area in enumerate(valid_targets):
                if to_area is not None and q_values[j] > best_q:
                    best_q = q_values[j]
                    best_action = (from_area, to_area)

        return best_action if best_action is not None else None

    def action_to_idx(self, action):
        return 0 if action is None else 1

    def reward_state(self, old_state, new_state, scale=1):
        player = old_state.player
        old_dice = old_state.player_num_dice[player]
        new_dice = new_state.player_num_dice[player]
        new_num_adjacent = new_state.player_max_size[player]
        old_num_adjacent = old_state.player_max_size[player]
        new_player_areas = new_state.player_areas
        old_player_areas = old_state.player_areas

        reward = 0
        if new_num_adjacent > old_num_adjacent:
            reward += 0.01 * (new_num_adjacent - old_num_adjacent)

        for i in range(len(new_state.player_num_dice)):
            if i != player and len(new_player_areas[i]) == 0 and len(old_player_areas[i]) != 0:
                reward += 0.2 * scale

        if new_dice > 0:
            reward += 0.01

        if new_state.winner == player and player != -1:
            reward += 1

        return reward
