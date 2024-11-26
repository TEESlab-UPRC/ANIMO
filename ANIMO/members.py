"""
    Copyright (C) 2024 Technoeconomics of Energy Systems laboratory - University of Piraeus Research Center (TEESlab-UPRC)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from mesa import Agent
import numpy as np
import random
import pandas as pd

"""Import input data"""
file_path = 'input_data.xlsx'
data = pd.read_excel(file_path)

'''Configure agents of the Member class'''

'''Values for agent-related parameters'''

# environmental concern
env_concern_mean = data.loc[data['Unnamed: 0'] == 'env_concern_mean', 'value'].iloc[0]
env_concern_std = data.loc[data['Unnamed: 0'] == 'env_concern_std', 'value'].iloc[0]

#  energy independence
nrg_independence_mean = data.loc[data['Unnamed: 0'] == 'nrg_independence_mean', 'value'].iloc[0]
nrg_independence_std = data.loc[data['Unnamed: 0'] == 'nrg_independence_std', 'value'].iloc[0]

# community sensitivity
community_sns_mean = data.loc[data['Unnamed: 0'] == 'community_sns_mean', 'value'].iloc[0]
community_sns_std = data.loc[data['Unnamed: 0'] == 'community_sns_std', 'value'].iloc[0]

# financial concern
financial_concern_mean = data.loc[data['Unnamed: 0'] == 'financial_concern_mean', 'value'].iloc[0]
financial_concern_std = data.loc[data['Unnamed: 0'] == 'financial_concern_std', 'value'].iloc[0]


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


class Member(Agent):
    def __init__(self, unique_id, model, agent_type="Community Member", scenario="Familiar"):
        super().__init__(unique_id, model)
        self.model = model
        self.agent_type = agent_type
        self.scenario = scenario
        self.env_concern = np.random.normal(env_concern_mean, env_concern_std)
        self.community_sns = np.random.normal(community_sns_mean, community_sns_std)
        self.fin_concern = np.random.normal(financial_concern_mean, financial_concern_std)
        self.nrg_independence = np.random.normal(nrg_independence_mean, nrg_independence_std)

        # Normalized agent-related parameters
        self.environmental_concern = abs(self.env_concern - env_concern_mean)
        self.financial_concern = abs(self.fin_concern - financial_concern_mean)
        self.sense_of_community = abs(self.community_sns - community_sns_mean)
        self.energy_independence = abs(self.nrg_independence - nrg_independence_mean)

        # Placeholder for typology / persona
        self.typology = None

        # Descriptive of ability to interact
        self.reach = custom_random(world_narrative=self.scenario)

        self.convincing_prowess = min(
            ((
                     self.environmental_concern - self.financial_concern + self.sense_of_community + self.energy_independence) / 4),
            1)
        self.connections_to_prospects = []
        self.categorize_typology()


    def categorize_typology(self):
        parameters = {
            'E': self.environmental_concern,
            'F': self.financial_concern,
            'S': self.sense_of_community,
            'I': self.energy_independence
        }
        # Sort parameters by values
        sorted_parameters = sorted(parameters.items(), key=lambda x: x[1], reverse=True)
        # Get the two largest parameters
        largest_parameters = sorted_parameters[:2]
        # Combine the first letter of each parameter to form the typology
        self.typology = ''.join(param[0] for param in largest_parameters)

        # Ensure only one combination of letters is considered
        if self.typology in ['SE', 'IF', 'IS', 'SF', 'IE']:
            self.typology = self.typology[::-1]  # Reverse the letters

        # Add 'Type ' prefix to typology
        self.typology = 'Type ' + self.typology

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
