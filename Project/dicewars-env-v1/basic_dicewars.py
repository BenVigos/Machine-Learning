from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import DefaultPlayer, AgressivePlayer, RandomPlayer, WeakerPlayerAttacker
from importlib import import_module



RENDER = True



PlayerX = import_module('playergroupX').Player() 

# a list of all participating Player objects
#players = [DefaultPlayer(), AgressivePlayer(), RandomPlayer(), WeakerPlayerAttacker()]
#players = [PlayerX, AgressivePlayer(), RandomPlayer(), WeakerPlayerAttacker()]
#players = [PlayerX, AgressivePlayer(), AgressivePlayer(),AgressivePlayer()]
players = [PlayerX,PlayerX,PlayerX,AgressivePlayer()]
playernames=[]
for i in range(len(players)):
    playernames.append(players[i].playername)

print(playernames)

# set up the game
game = Game(num_seats=len(players))
match = Match(game)

# # Instead of the above we can also load a previously saved match with the code below
# match = Match.load("filename")
#match = Match.load("savedmatch.save")
# # In case we would like to render this match again we might have to force drawing the board again
# # (the figure that is referred to in the match object probably no longer exists)
# match.drawboard()

# # Saving a specific match for later playback goes as follows
#match.save("savedmatch.save")


# Initialize the grid and state
grid, state = match.game.grid, match.state

# play the game until finsihed
while True:
    # get an action from the current player
    currentplayer = players[state.player]
    action = currentplayer.get_attack_areas(grid, state)
    
    grid, state = match.step(action)
#    print(match.state) 
    # render for graphical representation of gamestate
    if RENDER:
        match.render()
    
    # quit if game is finished
    if state.winner != -1:
        break

print(f"Winner: player {state.winner}, {players[state.winner].playername}")   
