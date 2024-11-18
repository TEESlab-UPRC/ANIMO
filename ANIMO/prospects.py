"""
Copyright (C) 2024 Technoeconomics of Energy Systems laboratory (TEESlab)
University of Piraeus Research Center (UPRC)

This code is free software: you can redistribute it and/ or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this code.  If not, see <https://www.gnu.org/licenses/>.

"""

from mesa import Agent, Model
import numpy as np
import random


# Supporting Function
#######################
def custom_random(world_narrative):
    # Adds functionality based on scenario / "world narrative"
    if world_narrative == "Familiar":
        return random.randint(1, 4)
    elif world_narrative == "Fragmented":
        if random.random() > 0.2:
            return random.randint(0, 2)
        else:
            return random.randint(3, 5)
    elif world_narrative == "Unified":
        if random.random() < 0.2:
            return random.randint(0, 2)
        else:
            return random.randint(3, 5)


class Prospect(Agent):
    def __init__(self, unique_id, model, agent_type="Prospective Member", scenario="Familiar"):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.scenario = scenario
        self.vision = custom_random(world_narrative=self.scenario)
        self.status = "Not Joined"
        self.receptivity_towards_innovation = None
        self.friends = []
        self.adopter_group = None
        self.number_of_friends = custom_random(world_narrative=self.scenario)
        self.attempts_to_be_convinced = 0

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
