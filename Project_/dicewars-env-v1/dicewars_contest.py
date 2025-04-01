'''
This script runs a contest between players. The results of the contest is printed to the console
You need to adapt the script in two places:

    1. change the list of playerfiles (line 20) to the proper filenames
        e.g.: 'playergroupX' -> 'playergroup23'

    2. change the constants if needed (probably only the NUMBER_OF_GAMES)

'''

# import required packages
from importlib import import_module
from dicewars.match import Match
from dicewars.game import Game
import sys

# ADAPT TO THE PLAYERS IN YOUR CONTEST
playerfiles = ['playergroupX', 'playergroupX', 'playergroupX', 'playergroupX']

# CONSTANTS
NUMBER_OF_GAMES = 2#2500  # The number of games that are played


RENDER = True#False  # if True, renders the gamestate, MUCH SLOWER!


# import the players that play against eachother
print()
print('======================================== IMPORTING PLAYERS ========================================')
print()

players = [import_module(playerfile).Player() for playerfile in playerfiles] 


# create an instance of scoreboard
playerscores = [0] * len(players)


# start the contest loop
print()
print('====================================== START CONTEST ==============================================')
print()

for gamenumber in range(NUMBER_OF_GAMES):
    
    # reset the game and generate new map
    game = Game(num_seats=len(players))
    match = Match(game)

    # play the game until finsihed
    
    # Initialize the state and info
    grid, state = match.game.grid, match.state
    
    while True:

        # get an action from the current player
        currentplayer = players[state.player]
        action = currentplayer.get_attack_areas(grid, state)

        grid, state = match.step(action)
        
        # render for graphical representation of gamestate
        if RENDER:
            match.render()

        # quit if game is finished
        if match.winner != -1:
            break
        
    # update scores
    winner = match.winner
    playerscores[winner] += 1

    # print the updated scores
    outp = f'\rgame:{gamenumber + 1}/{NUMBER_OF_GAMES} | '
    for name, score in zip(playerfiles, playerscores):
        outp += f'{name} : {score} | '
    print(outp, end="")
    sys.stdout.flush()

print()
print()
print('======================================  END CONTEST  ==============================================')
print()

