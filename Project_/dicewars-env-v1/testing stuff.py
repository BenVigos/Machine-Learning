import pickle
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import AgressivePlayer, RandomPlayer
from playergroupX import Player  # Import the agent

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 32
EPISODES = 100
EPSILON = 1  # Exploration factor
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95

RENDER = False


game_history = []

# Training loop
game = Game(num_seats=4)
match = Match(game)


state_size = 4  #simplest state, number of dice per player
action_size = len(game.area_num_dice) ** 2  # Assuming all possible (from, to) moves

agent = Player(state_size=state_size, action_size= action_size, MEMORY_SIZE= MEMORY_SIZE, EPSILON = EPSILON, LEARNING_RATE = LEARNING_RATE, BATCH_SIZE = BATCH_SIZE, GAMMA = GAMMA, EPSILON_MIN = EPSILON_MIN, EPSILON_DECAY = EPSILON_DECAY, model="dqn_model.keras")  # Our DQN player
players = [agent, AgressivePlayer(), RandomPlayer(), RandomPlayer()]


for episode in range(EPISODES):
    match = Match(Game(num_seats=4))
    player = match.player
    state = match.state
    done = False

    while not done and match.player != -1:
        player = match.player
        current_player = players[player]
        if player == 0:
            action = agent.get_attack_areas(match.game.grid, match.state)

            grid, new_state = match.step(action)

            reward = agent.reward_state(state, new_state)
            done = new_state.winner != -1

            valid_actions_next = agent.get_valid_actions(match.game.grid, new_state)

            agent.remember(state, action, reward, new_state, done, valid_actions_next)
            state = new_state
        else:
            action = current_player.get_attack_areas(match.game.grid, state)

            grid, state = match.step(action)

        if RENDER:
            match.render()

    agent.replay()

    game_history.append(match.state)
    print(f"Episode {episode + 1}/{EPISODES} - Epsilon: {agent.epsilon:.2f}")
    print(f"Victor was {match.winner}")

agent.model.save("dqn_model.keras")
# Save to a file
with open("game_history.pkl", "wb") as f:
    pickle.dump(game_history, f)
print("Training complete. Model saved!")
