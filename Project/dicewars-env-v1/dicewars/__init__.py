#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021 Thomas Schott <scotty@c-base.org>
#
# This file is part of dicewars.
#
# dicewars is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# dicewars is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with dicewars.  If not, see <http://www.gnu.org/licenses/>.

"""
DiceWars is a (quite addictive) quick game, inspired by the
`Risk <https://en.wikipedia.org/wiki/Risk_(game)>`_ board game.
It mixes strategy and luck, the objective is to conquer all areas on a
map by dice rolls.

It was published for browsers by GAMEDESIGN (2001 for
`Flash <https://www.gamedesign.jp/games/dicewars/flash/dice.html>`_,
then ported to `JavaScript <https://www.gamedesign.jp/games/dicewars/>`_).

**dicewars** implements the logic (the backend) that is required to create
playable game GUIs (the frontends) and (multi player) game servers in Python:

* generate random hexagonal cell grids (:class:`grid.Grid`)
* generate random match configurations (:class:`game.Game`)
* run matches according to the rules (:class:`match.Match`)
* implement and use AI players in matches
  (:class:`player.Player`, :class:`player.DefaultPlayer`)

Beside data for the game logic, :class:`grid.Grid` instances also provide
coordinates to render a map.
For frontends w/o point-in-polygon-test support :func:`util.pick_grid_area`
comes in handy for area selection.

.. _match-loop-example:

A minimal frontend match loop could look like this (AI players only):

.. code-block:: python

   from dicewars.match import Match
   from dicewars.player import DefaultPlayer

   match = Match()  # uses default grid and game (28x32 cells, 30 areas, 7 seats)
   ai_player = DefaultPlayer()

   while match.winner < 0:  # at least two players alive
       while match.winner < 0:  # do as many attacks as the current player wishes
           attack_areas = ai_player.get_attack_areas(match.game.grid, match.state)
           if attack_areas:  # tuple of from/to area indices -> attack!
               match.set_from_area(attack_areas[0])  # players's attacking area
               match.set_to_area(attack_areas[1])  # adjacent area to attack
               match.attack()  # roll dice, result is available in match.last_attack
               if match.player_num_areas[match.last_attack.to_player] == 0:
                   # attacked area was the owning player's last one -> eliminated
                   print(
                       f'player {match.last_attack.from_player} kicked out '
                       f'player {match.last_attack.to_player}.'
                   )
           else:  # None -> no more attacks, end player's turn
               break
       match.end_turn()  # supply dice to player's areas, advance to next player

   print(f'player {match.winner} wins the match!')

Instead of requesting ``attack_areas`` from ``ai_player``, just set the
from/to areas from user input for interactive matches.
"""
