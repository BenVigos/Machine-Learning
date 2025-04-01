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
Generate and run matches.

:class:`Match` instances manage and expose the current match state
required for the game logic. Additional (pre-calculated) data is
provided for AI players and frontend convenience.

A :class:`Match` is initialized from a match configuration
(a :class:`~dicewars.game.Game` instance). To restart a match, just
create a new instance from the same configuration.

The game logic is implemented in :meth:`Match.attack` and
:meth:`Match.end_turn`. Only these two methods change the match state.
All executed actions are available in :attr:`Match.history` for e.g.
match replays.

The match loop is:

* while at least 2 players are alive (:attr:`Match.winner` < `0`):

  * while the current player can and wants to attack:

    * set the attacking area (:meth:`Match.set_from_area`)
    * set the attacked area (:meth:`Match.set_to_area`)
    * roll dice to attack (:meth:`Match.attack`)

  * end the current player's turn (:meth:`Match.end_turn`)

The attacking/attacked areas choice may come from an AI player (e.g.
:class:`~dicewars.player.DefaultPlayer`) or from user input.
See :ref:`here <match-loop-example>` for a working code example.

For easy interfacing with AI players, frontends, (processing) libraries,
(network) protocols, etc., all :class:`Match`, :class:`State`,
:class:`Attack` and :class:`Supply` data are exposed as `int`, `tuple(int)`
or `tuple(tuple(int))` objects. All area/player references and parameters
are indices into the respective tuples.
"""

import pickle
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

from . game import Game
from . util import get_player_max_size


State = namedtuple(
    'State',
    'num_steps seat player winner area_players area_num_dice '
    'player_areas player_num_areas player_max_size player_num_dice player_num_stock'
)
"""
A convenience wrapper to access the current match state data at once. (`namedtuple`)

The (copied) value or (referenced) tuple properties of a :class:`Match`
instance are:

* :attr:`~Match.num_steps`
* :attr:`~Match.seat`
* :attr:`~Match.player`
* :attr:`~Match.winner`
* :attr:`~Match.area_players`
* :attr:`~Match.area_num_dice`
* :attr:`~Match.player_areas`
* :attr:`~Match.player_num_areas`
* :attr:`~Match.player_max_size`
* :attr:`~Match.player_num_dice`
* :attr:`~Match.player_num_stock`

The current State instance is available via :attr:`Match.state` and valid
until a (successful) call of :meth:`Match.attack` or :meth:`Match.end_turn`.

.. versionchanged:: 0.2.0
   Added ``num_steps``.
"""

Attack = namedtuple(
    'Attack',
    'step '
    'from_player from_area from_dice from_sum_dice '
    'to_player to_area to_dice to_sum_dice '
    'victory '
    'from_area_num_dice from_player_num_areas from_player_max_size from_player_num_dice '
    'to_area_num_dice to_player_num_areas to_player_max_size to_player_num_dice'
)
"""
Information and result of an executed attack. (`namedtuple`)

Attack instances are created in :meth:`Match.attack` and available via
:attr:`Match.last_attack` and :attr:`Match.history` afterwards.

.. attribute:: step
   :type: int

   The Attack's index in :attr:`Match.history`.

   .. versionadded:: 0.2.0

.. attribute:: from_player
   :type: int

   The index of the attacking player.

.. attribute:: from_area
   :type: int

   The index of the attacking area.

.. attribute:: from_dice
   :type: tuple(int)

   The randomly generated dice values for the attacking area.

.. attribute:: from_sum_dice
   :type: int

   The sum of :attr:`from_dice`.

.. attribute:: to_player
   :type: int

   The index of the attacked player.

.. attribute:: to_area
   :type: int

   The index of the attacked area.

.. attribute:: to_dice
   :type: tuple(int)

   The randomly generated dice values for the attacked area.

.. attribute:: to_sum_dice
   :type: int

   The sum of :attr:`to_dice`.

.. attribute:: victory
   :type: bool

   `True` if successful, `False` if defeated.

.. attribute:: from_area_num_dice
   :type: int

   The number of dice placed on the attacking area (always `1` for standard game rules).

   .. versionadded:: 0.2.0

.. attribute:: from_player_num_areas
   :type: int

   The total number of areas occupied by the attacking player.

   .. versionadded:: 0.2.0

.. attribute:: from_player_max_size
   :type: int

   The maximal number of adjacent areas occupied by the attacking player.

   .. versionadded:: 0.2.0

.. attribute:: from_player_num_dice
   :type: int

   The total number of dice placed on the attacking player’s areas.

   .. versionadded:: 0.2.0

.. attribute:: to_area_num_dice
   :type: int

   The number of dice placed on the attacked area.

   .. versionadded:: 0.2.0

.. attribute:: to_player_num_areas
   :type: int

   The total number of areas occupied by the attacked player.

   .. versionadded:: 0.2.0

.. attribute:: to_player_max_size
   :type: int

   The maximal number of adjacent areas occupied by the attacked player.

   .. versionadded:: 0.2.0

.. attribute:: to_player_num_dice
   :type: int

   The total number of dice placed on the attacked player’s areas.

   .. versionadded:: 0.2.0
"""

Supply = namedtuple('Supply', 'step player areas dice sum_dice area_num_dice player_num_stock')
"""
The outcome of dice supply at the end of a player's turn. (`namedtuple`)

Supply instances are created in :meth:`Match.end_turn` and available via
:attr:`Match.last_supply` and :attr:`Match.history` afterwards.

.. attribute:: step
   :type: int

   The Supply's index in :attr:`Match.history`.

   .. versionadded:: 0.2.0

.. attribute:: player
   :type: int

   The index of the player.

.. attribute:: areas
   :type: tuple(int)

   The indices of the player's areas that got dice supply.

.. attribute:: dice
   :type: tuple(int)

   The number of dice supplied to the areas in :attr:`areas`.

.. attribute:: sum_dice
   :type: int

   The sum of :attr:`dice`.

.. attribute:: area_num_dice
   :type: tuple(int)

   The number of dice placed on the areas in :attr:`areas`.

   .. versionadded:: 0.2.0

.. attribute:: player_num_stock
   :type: int

   The number of dice stored in the player's stock.

   .. versionchanged:: 0.2.0
      Renamed from "num_stock".
"""


class Match:
    AREA_MAX_NUM_DICE = Game.AREA_MAX_NUM_DICE
    """Maximal number of dice per area. (`int`)"""
    PLAYER_MAX_NUM_STOCK = 64
    """Maximal number of stored (i.e. not supplied to areas) dice per player. (`int`)"""

    def __init__(self, game=None):
        """
        Generate a runnable match.

        :param game: :class:`~dicewars.game.Game` instance used to initialize
           the match state (if `None`: a :class:`~dicewars.game.Game` with
           default parameters is generated)
        :type game: Game or None
        :raise TypeError: if ``game`` is not a `Game` instance
        """

        if game is None:
            self._game = Game()
        else:
            if not isinstance(game, Game):
                raise TypeError('game must be an instance of Game')
            self._game = game

        # internal areas/players states
        self.__area_players = list(self._game.area_seats)
        self.__area_num_dice = list(self._game.area_num_dice)
        self.__player_areas = list(list(p_areas) for p_areas in self._game.seat_areas)
        self.__player_num_areas = list(self._game.seat_num_areas)
        self.__player_max_size = list(self._game.seat_max_size)
        self.__player_num_dice = list(self._game.seat_num_dice)
        self.__player_num_stock = [0] * self._game.num_seats

        # exposed (read only) mirrors of internal areas/players states
        self._area_players = self._game.area_seats
        self._area_num_dice = self._game.area_num_dice
        self._player_areas = self._game.seat_areas
        self._player_num_areas = self._game.seat_num_areas
        self._player_max_size = self._game.seat_max_size
        self._player_num_dice = self._game.seat_num_dice
        self._player_num_stock = tuple(self.__player_num_stock)

        self._seat_idx = 0 if 1 < self._game.num_seats else -1
        self._winner = -1 if self._seat_idx != -1 else 0
        self._from_area_idx = -1
        self._to_area_idx = -1
        self._last_attack = None
        self._last_supply = None

        self.__history = []
        self._history = None  # exposed (read only) mirror, created/updated only on request

        self._state = None  # for convenient full match state access/passing
        self._update_state()

    @property
    def game(self):
        """The :class:`~dicewars.game.Game` instance (configuration) used for the match."""
        return self._game

    @property
    def state(self):
        """The :class:`State` instance of the current match state."""
        return self._state

    @property
    def num_steps(self):
        """
        The number of (successful) :meth:`attack` and :meth:`end_turn` calls. (`int`)

        .. versionadded:: 0.2.0
        """

        return len(self.__history)

    @property
    def seat(self):
        """The current player seat index, `-1` if match is finished. (`int`)"""
        return self._seat_idx

    @property
    def player(self):
        """The current player's index, `-1` if match is finished. (`int`)"""
        return self._game.seat_order[self._seat_idx] if self._seat_idx != -1 else -1

    @property
    def winner(self):
        """The last remaining player's index, `-1` if match is not finished.  (`int`)"""
        return self._winner

    @property
    def area_players(self):
        """The occupying player's index for each area. (`tuple(int)`)"""
        return self._area_players

    @property
    def area_num_dice(self):
        """The number of dice placed on each area. (`tuple(int)`)"""
        return self._area_num_dice

    @property
    def player_areas(self):
        """The indices of all areas occupied by each player. (`tuple(tuple(int))`)"""
        return self._player_areas

    @property
    def player_num_areas(self):
        """The total number of areas occupied by each player. (`tuple(int)`)"""
        return self._player_num_areas

    @property
    def player_max_size(self):
        """The maximal number of adjacent areas occupied by each player. (`tuple(int)`)"""
        return self._player_max_size

    @property
    def player_num_dice(self):
        """The total number of dice placed on each player's areas. (`tuple(int)`)"""
        return self._player_num_dice

    @property
    def player_num_stock(self):
        """The number of each player's stored dice that could not be supplied to areas. (`tuple(int)`)"""
        return self._player_num_stock

    @property
    def from_area(self):
        """The index of the currently set attacking area, `-1` if not set. (`int`)"""
        return self._from_area_idx

    @property
    def to_area(self):
        """The index of the currently set attacked area, `-1` if not set. (`int`)"""
        return self._to_area_idx

    @property
    def last_attack(self):
        """The :class:`Attack` instance created by the last (successful) call of :meth:`attack`."""
        return self._last_attack

    @property
    def last_supply(self):
        """The :class:`Supply` instance created by the last (successful) call of :meth:`end_turn`."""
        return self._last_supply

    @property
    def history(self):
        r"""
        The sequence of all :class:`Attack`\s and :class:`Supply`\s so far. (`tuple(Attack/Supply)`)

        .. versionadded:: 0.2.0
        """

        if self._history is None:
            self._history = tuple(self.__history)
        return self._history

    def set_from_area(self, area_idx):
        """
        Validate and set or unset the attacking area.

        :param int area_idx: index of the attacking area, < `0` to unset
        :return: `True` if accepted and changed or unset,
           `False` when rejected or unchanged
        :rtype: bool
        :raise TypeError: if ``area_idx`` is not `int`
        """

        if not isinstance(area_idx, int):
            raise TypeError('area_idx must be int')

        if self._seat_idx == -1:
            return False
        if len(self.__area_players) <= area_idx:
            return False

        if area_idx < 0:  # unset
            if self._from_area_idx == -1:
                return False
            self._from_area_idx = -1
            return True
        if self._from_area_idx == area_idx:
            return False

        from_player_idx = self.player
        if from_player_idx != self.__area_players[area_idx]:
            return False
        assert area_idx in self.__player_areas[from_player_idx]
        assert 0 < self.__area_num_dice[area_idx]
        if self.__area_num_dice[area_idx] == 1:
            return False
        if self._to_area_idx != -1:
            if self._to_area_idx not in self._game.grid.areas[area_idx].neighbors:
                return False
            assert area_idx in self._game.grid.areas[self._to_area_idx].neighbors
            assert from_player_idx != self.__area_players[self._to_area_idx]

        self._from_area_idx = area_idx
        return True

    def set_to_area(self, area_idx):
        """
        Validate and set or unset the attacked area.

        :param int area_idx: index of the attacked area, < `0` to unset
        :return: `True` if accepted and changed or unset,
           `False` when rejected or unchanged
        :rtype: bool
        :raise TypeError: if ``area_idx`` is not `int`
        """

        if not isinstance(area_idx, int):
            raise TypeError('area_idx must be int')

        if self._seat_idx == -1:
            return False
        if len(self.__area_players) <= area_idx:
            return False

        if area_idx < 0:  # unset
            if self._to_area_idx == -1:
                return False
            self._to_area_idx = -1
            return True
        if self._to_area_idx == area_idx:
            return False

        to_player_idx = self.__area_players[area_idx]
        if self.player == to_player_idx:
            return False
        assert area_idx in self.__player_areas[to_player_idx]
        assert 0 < self.__area_num_dice[area_idx]
        if self._from_area_idx != -1:
            if self._from_area_idx not in self._game.grid.areas[area_idx].neighbors:
                return False
            assert area_idx in self._game.grid.areas[self._from_area_idx].neighbors
            assert 1 < self.__area_num_dice[self._from_area_idx]

        self._to_area_idx = area_idx
        return True

    def attack(self):
        """
        Validate and execute an attack for the current player.

        The attack is executed only when valid attacking/attacked areas
        have been set before. The attack's result is available via
        :attr:`last_attack`. Attacking/attacked areas are unset after
        execution.

        :return: `True` if executed and match state is updated,
           `False` when rejected (match state has not changed)
        :rtype: bool
        """

        self._last_attack = None
        if self._seat_idx == -1:
            return False
        if self._from_area_idx == -1 or self._to_area_idx == -1:
            return False

        from_player_idx = self.player
        from_player_areas = self.__player_areas[from_player_idx]
        to_player_idx = self.__area_players[self._to_area_idx]
        to_player_areas = self.__player_areas[to_player_idx]
        assert from_player_idx == self.player
        assert from_player_idx == self.__area_players[self._from_area_idx]
        assert from_player_idx != to_player_idx
        assert to_player_idx == self.__area_players[self._to_area_idx]
        assert self._from_area_idx in from_player_areas
        assert self._from_area_idx not in to_player_areas
        assert self._from_area_idx in self._game.grid.areas[self._to_area_idx].neighbors
        assert self._to_area_idx in to_player_areas
        assert self._to_area_idx not in from_player_areas
        assert self._to_area_idx in self._game.grid.areas[self._from_area_idx].neighbors

        from_num_dice = self.__area_num_dice[self._from_area_idx]
        from_rand_dice = tuple(random.randint(1, 6) for _ in range(from_num_dice))
        from_sum_dice = sum(from_rand_dice)
        to_num_dice = self.__area_num_dice[self._to_area_idx]
        to_rand_dice = tuple(random.randint(1, 6) for _ in range(to_num_dice))
        to_sum_dice = sum(to_rand_dice)
        assert 1 < from_num_dice
        assert 0 < to_num_dice
        assert from_num_dice <= from_sum_dice
        assert to_num_dice <= to_sum_dice

        attack_num_dice = from_num_dice - 1
        self.__area_num_dice[self._from_area_idx] = 1
        victory = to_sum_dice < from_sum_dice
        if victory:
            self.__area_players[self._to_area_idx] = from_player_idx
            self._area_players = tuple(self.__area_players)
            from_player_areas.append(self._to_area_idx)
            to_player_areas.remove(self._to_area_idx)
            self._player_areas = tuple(tuple(p_areas) for p_areas in self.__player_areas)
            self.__player_num_areas[from_player_idx] = len(from_player_areas)
            self.__player_num_areas[to_player_idx] = len(to_player_areas)
            self._player_num_areas = tuple(self.__player_num_areas)
            self.__player_max_size[from_player_idx] = get_player_max_size(self._game.grid.areas, from_player_areas)
            self.__player_max_size[to_player_idx] = get_player_max_size(self._game.grid.areas, to_player_areas)
            self._player_max_size = tuple(self.__player_max_size)
            self.__area_num_dice[self._to_area_idx] = attack_num_dice
            self.__player_num_dice[to_player_idx] -= to_num_dice
            assert self.__player_num_areas[to_player_idx] <= self.__player_num_dice[to_player_idx]
            if self.__player_num_areas[from_player_idx] == len(self._game.grid.areas):
                self._seat_idx = -1
                self._winner = from_player_idx
        else:
            self.__player_num_dice[from_player_idx] -= attack_num_dice
            assert self.__player_num_areas[from_player_idx] <= self.__player_num_dice[from_player_idx]
        self._area_num_dice = tuple(self.__area_num_dice)
        self._player_num_dice = tuple(self.__player_num_dice)

        self._last_attack = Attack(
            self.num_steps,
            from_player_idx, self._from_area_idx, from_rand_dice, from_sum_dice,
            to_player_idx, self._to_area_idx, to_rand_dice, to_sum_dice,
            victory,
            self.__area_num_dice[self._from_area_idx], self.__player_num_areas[from_player_idx],
            self.__player_max_size[from_player_idx], self.__player_num_dice[from_player_idx],
            self.__area_num_dice[self._to_area_idx], self.__player_num_areas[to_player_idx],
            self.__player_max_size[to_player_idx], self.__player_num_dice[to_player_idx],
        )
        self.__history.append(self._last_attack)
        self._history = None

        self._update_state()
        self._from_area_idx = -1
        self._to_area_idx = -1
        return True

    def end_turn(self):
        """
        End current player's turn and advance to the next player.

        The player's :attr:`player_max_size` number of dice is randomly
        supplied to the player's areas (or stored). The outcome is available
        via :attr:`last_supply`. The player on the next seat becomes the
        current player.

        :return: `True` if match state is updated,
           `False` when the match is finished already
        :rtype: bool
        """

        self._last_supply = None
        if self._seat_idx == -1:
            return False

        player_idx = self.player
        num_stock = self.__player_num_stock[player_idx] + self.__player_max_size[player_idx]
        assert num_stock
        if self.PLAYER_MAX_NUM_STOCK < num_stock:
            num_stock = self.PLAYER_MAX_NUM_STOCK

        player_areas = self.__player_areas[player_idx]
        area_supplies = dict((a_idx, 0) for a_idx in player_areas)
        while num_stock:
            areas = [
                a_idx for a_idx in player_areas
                if self.__area_num_dice[a_idx] < self.AREA_MAX_NUM_DICE
            ]
            if areas:
                area_idx = random.choice(areas)
                self.__area_num_dice[area_idx] += 1
                self.__player_num_dice[player_idx] += 1
                num_stock -= 1
                area_supplies[area_idx] += 1
            else:
                break
        self._area_num_dice = tuple(self.__area_num_dice)
        self._player_num_dice = tuple(self.__player_num_dice)
        self.__player_num_stock[player_idx] = num_stock
        self._player_num_stock = tuple(self.__player_num_stock)

        area_supplies = tuple(
            (a_idx, n_dice, self.__area_num_dice[a_idx])
            for a_idx, n_dice in area_supplies.items() if n_dice
        )
        self._last_supply = Supply(
            self.num_steps, player_idx,
            tuple(area_supply[0] for area_supply in area_supplies),
            tuple(area_supply[1] for area_supply in area_supplies),
            sum(area_supply[1] for area_supply in area_supplies),
            tuple(area_supply[2] for area_supply in area_supplies),
            num_stock
        )
        self.__history.append(self._last_supply)
        self._history = None

        while True:
            self._seat_idx += 1
            if self._seat_idx == self._game.num_seats:
                self._seat_idx = 0
            if self.__player_num_areas[self.player]:
                assert self.__player_num_areas[self.player] < len(self._game.grid.areas)
                break

        self._update_state()
        self._from_area_idx = -1
        self._to_area_idx = -1
        return True
    
    def step(self, attack_areas):
        """
        Execute the attack defined by attack_areas.
        If attack_areas is None or attack is invalid, end the turn

        :param tuple(int, int) attack_areas: index (from, to) which the attack is
        :return: The state after the attack
        :rtype: State
        """
        # pass the action
        if attack_areas:  # tuple of from/to area indices -> attack!
            self.set_from_area(attack_areas[0])  # players's attacking area
            self.set_to_area(attack_areas[1])  # adjacent area to attack
            legal_move = self.attack()
            if not legal_move:
                self.end_turn()
    
        else:   # None -> no more attacks, end currentplayer's turn
            self.end_turn()
            
        return self.game.grid, self.state

    def _update_state(self):
        self._state = State(
            self.num_steps, self._seat_idx, self.player, self._winner,
            self._area_players, self._area_num_dice,
            self._player_areas, self._player_num_areas, self._player_max_size,
            self._player_num_dice, self._player_num_stock
        )
        
    def render(self):
        if not hasattr(self, "_fig"):
            self.drawboard()
        
        
        self._ax.set_title(f"Moves played: {self.num_steps : >5}")
        for i in range(len(self._game.grid.areas)):
            self._area_patches[i].set_facecolor(self._player_colors[self.area_players[i]])
            self._area_numbers[i].set_text(str(self.area_num_dice[i]))
            
        
        # force drawing the board
        self._fig.canvas.draw()  
        self._fig.canvas.flush_events()   
           
        
        
    def drawboard(self):
        self._player_colors = ['salmon', 'lightgreen', 'skyblue', 'wheat']

        # setup the figure
        plt.rcParams['toolbar'] = 'None'
        xmax, ymax = self._game.grid._map_size
        self._fig, self._ax = plt.subplots(num="Dicewars", figsize=(8,8), facecolor='lightgray', tight_layout=True)
        self._ax.set_xlim(-10, xmax+10)
        self._ax.set_ylim(-10, ymax+10)
        self._ax.axis('off')
        self._ax.set_aspect('equal', adjustable='box', anchor='C')
        self._ax.set_title(f"Moves played: {self.num_steps : >5}")
        
        cells = self._game.grid.cells
        grid_w, grid_h = self._game.grid.grid_size
        
        hex_radius = 0.1
        x_offset = hex_radius * 3**0.5
        y_offset = hex_radius*(1+3**0.5)/2
        
        self._area_patches = dict()
        self._area_numbers = dict()
        legend_elements = [patches.Patch(color=self._player_colors[i], label=f'Player {i}') for i in range(self._game.num_seats)]
        self._ax.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))
        
        for area in self._game.grid.areas:
            verts = area.border + (area.border[0],)
            patch = patches.PathPatch(Path(verts), facecolor='lightgray')
            self._ax.add_patch(patch)
            self._area_patches[area.idx] = patch
            
            center_cell = self._game.grid.cells[area.center]
            bbox = center_cell.bbox
            dx = (bbox[1][0]-bbox[0][0])/2
            dy = (bbox[1][1]-bbox[0][1])/2
            self._area_numbers[area.idx] = self._ax.text((bbox[0][0])+dx, bbox[0][1]+dy, "", 
                                                         horizontalalignment='center',
                                                         verticalalignment='center')

        # force drawing the board
        self._fig.canvas.draw()  
        self._fig.canvas.flush_events()  
        self._fig.show() 
    
    
    def save(self, filename: str):
        """
        save a match to filename
        example: match.save("match1.txt")
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Class instance saved to {filename} successfully.")
        except Exception as e:
            print(f"Error saving to file: {e}")

    @classmethod
    def load(cls, filename):
        """
        You can load a match instance by using:
          match = Match.load("filename")
        """
        try:
            with open(filename, 'rb') as file:
                loaded_instance = pickle.load(file)
            print(f"Class instance loaded from {filename} successfully.")
            return loaded_instance
        except Exception as e:
            print(f"Error loading from file: {e}")
            return None