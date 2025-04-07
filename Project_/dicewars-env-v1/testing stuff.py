import pickle
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import AgressivePlayer, RandomPlayer
from playergroupLocal import Player  # Import the agent
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Training Loss")
smooth_line, = ax.plot([], [], label="Moving Average", linestyle="--", color="orange")
ax.set_xlabel("Actions")
ax.set_ylabel("Loss")
ax.set_yscale('log')
ax.set_title("DQN Loss over Time")
ax.grid(True)
ax.legend()

# Moving average window
SMOOTHING_WINDOW = 50  # You can tweak this number

def update_plot(loss_history):
    # Update raw loss line
    x_data = np.arange(len(loss_history)) * TRAIN_AFTER_ACTIONS
    line.set_xdata(x_data)
    line.set_ydata(loss_history)

    # Compute moving average
    if len(loss_history) >= SMOOTHING_WINDOW:
        smoothed = np.convolve(loss_history, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='valid')
        smooth_x = x_data[SMOOTHING_WINDOW-1:]  # Match x-data to smoothed array
        smooth_line.set_xdata(smooth_x)
        smooth_line.set_ydata(smoothed)
    else:
        smooth_line.set_xdata([])
        smooth_line.set_ydata([])

    # Update plot
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


losses = []


# # Plot of max adjacent fields averaged over 10 steps
# # Create a new figure for adjacency ratio
# adj_fig, adj_ax = plt.subplots()
# adj_line, = adj_ax.plot([], [], label="Adjacency Ratio", linestyle="--", color="green")
#
# adj_ax.set_xlabel("Steps")
# adj_ax.set_ylabel("Adjacency Ratio")
# adj_ax.set_ylim(0, 1)  # Ratio is between 0 and 1
# adj_ax.set_title("Average Adjacency Ratio Over Time")
# adj_ax.grid(True)
# adj_ax.legend()
#
# adj_ratio_history = []
#
# def update_adjacency_plot(new_state, step, player):
#     global adj_ratio_history
#
#     max_adjacent = new_state.player_max_size[player]
#     num_areas = new_state.player_num_areas[player]
#
#     # Prevent division by zero
#     if num_areas > 0:
#         ratio = max_adjacent / num_areas
#     else:
#         ratio = 0.0
#
#     adj_ratio_history.append(ratio)
#
#     # Only update every 10 steps
#     if step % 10 == 0:
#         x_data = np.arange(len(adj_ratio_history)) * 10
#         adj_line.set_xdata(x_data)
#         adj_line.set_ydata(adj_ratio_history)
#         adj_ax.relim()
#         adj_ax.autoscale_view()
#         adj_fig.canvas.draw()
#         adj_fig.canvas.flush_events()





# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.01
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPISODES = 100
EPSILON = 0  # Exploration factor
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99995
TRAIN_AFTER_ACTIONS = 10
UPDATE_TARGET = 1000
MAX_STEPS = np.inf
steps = 0

SCALE = 1

DIM_FACTOR = 1

RENDER = False

game_history = []
reward_history = []
loss_history = []
victory_history = []
testing_history = []


# Training loop
game = Game(num_seats=4)
match = Match(game)

state_size = 22  # simplest state, number of dice per player
action_size = 10  # Assuming all possible (from, to) moves

agent = Player(state_size=state_size, action_size=action_size, MEMORY_SIZE=MEMORY_SIZE, EPSILON=EPSILON,
               LEARNING_RATE=LEARNING_RATE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, EPSILON_MIN=EPSILON_MIN,
               EPSILON_DECAY=EPSILON_DECAY)  # Our DQN player
players = [agent, RandomPlayer(), RandomPlayer(), RandomPlayer()]


def test_agent(num_games, game=None, RENDER = False):
    wins = 0


    if game == None:
        game = Game(num_seats=4)

    for i in range(num_games):
        match = Match(game)

        grid, state = match.game.grid, match.state

        while True:
            currentplayer = players[state.player]

            action = currentplayer.get_attack_areas(grid, state)

            grid, state = match.step(action)

            if RENDER:
                match.render()

            if state.winner != -1 or state.player_num_dice[0] == 0:
                if state.winner == 0:
                    print("Win")
                    wins += 1
                break
    return wins/num_games



for episode in range(EPISODES):
    match = Match(game)
    player = match.player
    state = match.state
    done = False

    total_reward = 0
    total_loss = 0

    if steps >= MAX_STEPS:
        break

    while not done and match.player != -1 and match.player_num_dice[0] > 0:

        player = match.player
        current_player = players[player]
        if player == 0:
            steps += 1
            action = agent.get_attack_areas(match.game.grid, match.state)

            grid, new_state = match.step(action)

            reward = agent.reward_state(state, new_state, SCALE)
            done = new_state.winner != -1

            valid_actions_next = agent.get_valid_actions(match.game.grid, new_state)

            agent.remember(grid, state, action, reward, new_state, done, valid_actions_next)
            state = new_state

            total_reward += reward

            if steps % TRAIN_AFTER_ACTIONS == 0 and steps > BATCH_SIZE:
                loss = agent.replay()

                if loss is not None:
                    total_loss += loss
                    losses.append(loss)
                    if len(losses) % 10 == 0:
                        update_plot(losses)

            if steps % UPDATE_TARGET == 0:
                agent.update_target()

        else:
            action = current_player.get_attack_areas(match.game.grid, state)

            grid, state = match.step(action)

        if RENDER:
            match.render()

    if match.state.winner == 0:
        victory_history.append(1)
    else:
        victory_history.append(0)
    reward_history.append(total_reward)
    game_history.append(match)
    loss_history.append(total_loss/(steps/TRAIN_AFTER_ACTIONS))

    print(
        f"Episode {episode + 1}/{EPISODES} - Epsilon: {agent.epsilon:.2f} \n Reward: {total_reward} | Steps {steps} \n")

    # if episode % 50 == 0:
        # win_rate = test_agent(num_games=10, game = game)
        #
        # print(f"TEST: Win rate {win_rate}")
        # testing_history.append(win_rate)

    SCALE *= DIM_FACTOR

agent.model.save("dqn_model_1_hidden_layer-64.keras")

print("Training complete. Model saved!")

# Save to a file
training_data = {
    "game_history": game_history,
    "reward_history": reward_history,
    "loss_history": loss_history,
    "victory_history": victory_history,
    "testing_history": testing_history,
    "losses": losses,
}

# Save to a pickle file
with open("training_data_4.pkl", "wb") as f:
    pickle.dump(training_data, f)


print("Training data saved!")