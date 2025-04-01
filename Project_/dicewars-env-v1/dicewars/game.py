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
Generate random match configurations.

Each :class:`~dicewars.grid.Area` of a :class:`~dicewars.grid.Grid`
is assigned to a random player seat and gets a random supply of about
three dice. The seat order (i.e. the sequence of the player's turns)
is randomly shuffled.

All area and seat data are exposed as tuples. Area and seat references
are indices into these tuples.

:class:`Game` instances are immutable and intended to generate
runnable :class:`~dicewars.match.Match` instances.
"""

import random

from . grid import Grid
from . util import get_player_max_size
import pickle


class Game:
    DEFAULT_NUM_SEATS = 4
    """Default for the ``num_seats`` parameter. (`int`)"""
    AREA_MAX_NUM_DICE = 8
    """Maximal number of dice per area. (`int`)"""

    def __init__(self, grid=None, num_seats=DEFAULT_NUM_SEATS):
        """
        Generate a game (assign seat/dice to areas, shuffle seats and calculate seat states).

        :param grid: :class:`~dicewars.grid.Grid` instance that provides the
           areas to use (if `None`: a :class:`~dicewars.grid.Grid` with
           default parameters is generated)
        :type grid: Grid or None
        :param int num_seats: number of player seats
        :raise TypeError: if ``grid`` is not a `Grid` instance or ``num_seats`` is not `int`
        :raise ValueError: if ``num_seats`` is out of range
        """

        if grid is None:
            self._grid = Grid()
        else:
            if not isinstance(grid, Grid):
                raise TypeError('grid must be an instance of Grid')
            self._grid = grid

        num_areas = len(self._grid.areas)

        if not isinstance(num_seats, int):
            raise TypeError('num_seats must be int')
        if num_seats < 1:
            raise ValueError('num_seats must be > 0')
        if num_areas < num_seats:
            raise ValueError(f'num_seats must be <= grid.num_areas={num_areas}')

        seat_random_order = [s_idx for s_idx in range(num_seats)]

        # random seat order
        random.shuffle(seat_random_order)
        self._seat_order = tuple(seat_random_order)

        # random area-to-seat assignments
        random.shuffle(seat_random_order)
        self._area_seats = [seat_random_order[a_idx % num_seats] for a_idx in range(num_areas)]
        random.shuffle(self._area_seats)
        self._area_seats = tuple(self._area_seats)

        # random dice-to-area assignments (~3 dice per area)
        self._area_num_dice = [1] * num_areas
        random.shuffle(seat_random_order)
        random_idx = 0
        # TODO: equal number of dice per seat?
        for _ in range(num_areas * 2):
            seat_idx = seat_random_order[random_idx]
            areas = [
                a_idx for a_idx in range(num_areas)
                if self._area_seats[a_idx] == seat_idx and self._area_num_dice[a_idx] < self.AREA_MAX_NUM_DICE
            ]
            if areas:
                self._area_num_dice[random.choice(areas)] += 1
            random_idx += 1
            if random_idx == num_seats:
                random_idx = 0
        self._area_num_dice = tuple(self._area_num_dice)

        # seat status calculations
        self._seat_areas = tuple(
            tuple(a_idx for a_idx in range(num_areas) if self._area_seats[a_idx] == s_idx)
            for s_idx in range(num_seats)
        )
        self._seat_num_areas = tuple(len(s_areas) for s_areas in self._seat_areas)
        self._seat_max_size = tuple(
            get_player_max_size(self._grid.areas, self._seat_areas[seat_idx])
            for seat_idx in range(num_seats)
        )
        self._seat_num_dice = tuple(
            sum(s_dice) for s_dice in (
                (self._area_num_dice[a_idx] for a_idx in self._seat_areas[s_idx])
                for s_idx in range(num_seats)
            )
        )

    @property
    def grid(self):
        """The :class:`~dicewars.grid.Grid` instance used for the game."""
        return self._grid

    @property
    def num_seats(self):
        """The number of player seats. (`int`)"""
        return len(self._seat_order)

    @property
    def seat_order(self):
        """The player indices in the order the players will do their turns. (`tuple(int)`)"""
        return self._seat_order

    @property
    def area_seats(self):
        """The seat indices assigned to each area. (`tuple(int)`)"""
        return self._area_seats

    @property
    def area_num_dice(self):
        """The number of dice supplied to each area. (`tuple(int)`)"""
        return self._area_num_dice

    @property
    def seat_areas(self):
        """The area indices assigned to each seat. (`tuple(tuple(int))`)"""
        return self._seat_areas

    @property
    def seat_num_areas(self):
        """The total number of areas assigned to each seat. (`tuple(int)`)"""
        return self._seat_num_areas

    @property
    def seat_max_size(self):
        """The maximal number of adjacent areas assigned to each seat. (`tuple(int)`)"""
        return self._seat_max_size

    @property
    def seat_num_dice(self):
        """The total number of dice supplied to each seat's areas. (`tuple(int)`)"""
        return self._seat_num_dice
    
    def save(self, filename: str):
        """
        save a game to filename
        example: game.save("game.txt")
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
        You can load a game instance by using:
          game = Game.load("filename")
        """
        try:
            with open(filename, 'rb') as file:
                loaded_instance = pickle.load(file)
            print(f"Class instance loaded from {filename} successfully.")
            return loaded_instance
        except Exception as e:
            print(f"Error loading from file: {e}")
            return None