import keras
import pickle
from matplotlib import pyplot as plt
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import AgressivePlayer, RandomPlayer
from playergroupLocal import Player  # Import the agent
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np



def test_agent(num_games, game=None, RENDER = False):
    wins = 0


    if game == None:
        game = Game(num_seats=4)

    for i in range(num_games):
        print(i)
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


agent = Player(EPSILON=0, EPSILON_MIN=0, model="Project_/dicewars-env-v1/dqn_model_1_hidden_layer-64-500_lower_eps.keras")  # Our DQN player
players = [agent, RandomPlayer(), RandomPlayer(), RandomPlayer()]

stat = test_agent(num_games=50, RENDER=True)
print(stat)