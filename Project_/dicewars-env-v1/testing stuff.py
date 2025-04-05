import pickle
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import AgressivePlayer, RandomPlayer
from playergroupX import Player  # Import the agent
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Training Loss")
ax.set_xlabel("Actions")
ax.set_ylabel("Loss")
ax.set_yscale('log')
ax.set_title("DQN Loss over Time")
ax.grid(True)
ax.legend()


def update_plot(loss_history):
    line.set_xdata(np.arange(len(loss_history))*TRAIN_AFTER_ACTIONS)
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


losses = []

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.01
MEMORY_SIZE = 10000
BATCH_SIZE = 128
EPISODES = 100
EPSILON = 1  # Exploration factor
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
TRAIN_AFTER_ACTIONS = 8
UPDATE_TARGET = 1000
steps = 0

RENDER = False

game_history = []

# Training loop
game = Game(num_seats=4)
match = Match(game)

state_size = 210  # simplest state, number of dice per player
action_size = len(game.area_num_dice) ** 2  # Assuming all possible (from, to) moves

agent = Player(state_size=state_size, action_size=action_size, MEMORY_SIZE=MEMORY_SIZE, EPSILON=EPSILON,
               LEARNING_RATE=LEARNING_RATE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, EPSILON_MIN=EPSILON_MIN,
               EPSILON_DECAY=EPSILON_DECAY)  # Our DQN player
players = [agent, AgressivePlayer(), AgressivePlayer(), AgressivePlayer()]

for episode in range(EPISODES):
    match = Match(game)
    player = match.player
    state = match.state
    done = False

    total_reward = 0

    while not done and match.player != -1 and match.player_num_dice[0] > 0:

        player = match.player
        current_player = players[player]
        if player == 0:
            steps += 1
            action = agent.get_attack_areas(match.game.grid, match.state)

            grid, new_state = match.step(action)

            reward = agent.reward_state(state, new_state)
            done = new_state.winner != -1

            valid_actions_next = agent.get_valid_actions(match.game.grid, new_state)

            agent.remember(grid, state, action, reward, new_state, done, valid_actions_next)
            state = new_state

            total_reward += reward

            if steps % TRAIN_AFTER_ACTIONS == 0 and steps > BATCH_SIZE:
                loss = agent.replay()

                if loss is not None:
                    losses.append(loss)
                    if len(losses) % 50 == 0:
                        update_plot(losses)

            if steps % UPDATE_TARGET == 0:
                agent.update_target()

        else:
            action = current_player.get_attack_areas(match.game.grid, state)

            grid, state = match.step(action)

        if RENDER:
            match.render()

    game_history.append(match.state)
    print(
        f"Episode {episode + 1}/{EPISODES} - Epsilon: {agent.epsilon:.2f} \n Reward: {total_reward} | Steps {steps} \n")

agent.model.save("dqn_model.keras")
# Save to a file
with open("game_history.pkl", "wb") as f:
    pickle.dump(game_history, f)
print("Training complete. Model saved!")
