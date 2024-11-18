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
from mesa.space import MultiGrid
from mesa.time import RandomActivationByType
from members import Member
from prospects import Prospect
import numpy as np
import random
from mesa.datacollection import DataCollector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class EnergyCommunityModel(Model):
    def __init__(self, num_members, num_prospects, width, height, scenario="Familiar"):
        self.width = width
        self.height = height
        self.scenario = scenario
        self.num_members = num_members
        self.num_prospects = num_prospects
        self.new_members_count = 0  # Track the number of new members
        self.new_members_count_list = []  # List to store counts
        self.percentage_list = []  # List to store percentages
        self.num_attempts = 0
        """A physical world to place agents in"""
        self.grid = MultiGrid(self.width, self.height, torus=True)

        """Initiate activation schedule"""
        self.schedule = RandomActivationByType(self)
        self.running = True

        # Create members
        for i in range(self.num_members):
            member = Member(i, self, scenario=self.scenario)
            self.schedule.add(member)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(member, (x, y))

        # Create prospects
        for i in range(self.num_prospects):
            prospect = Prospect(i + self.num_members, self, scenario=self.scenario)
            self.schedule.add(prospect)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(prospect, (x, y))

        self.adopter_group_assignment()
        self.prospects_find_peers()

        # Collect data
        self.datacollector = DataCollector(
            model_reporters={
                "New Members": self.collect_new_members,
                "Growth Percentage": self.collect_growth_percentage
            }
        )

    def adopter_group_assignment(self):
        #  Applying the "Diffusion of Innovation theory" to agents (instances) of the Prospect class, also considering "world narratives"
        prospects = [agent for agent in self.schedule.agents if isinstance(agent, Prospect)]

        scenario = self.scenario
        if scenario == "Familiar":
            innovators_number = int(0.025 * len(prospects))
            early_adopters_number = int(0.135 * len(prospects))
            early_majority_number = int(0.34 * len(prospects))
            late_majority_number = int(0.34 * len(prospects))
            laggards_number = int(0.16 * len(prospects))

            innovators_list = prospects[:innovators_number]
            early_adopters_list = prospects[innovators_number:(innovators_number + early_adopters_number)]
            early_majority_list = prospects[(innovators_number + early_adopters_number):(
                    innovators_number + early_adopters_number + early_majority_number)]
            late_majority_list = prospects[
                                 (innovators_number + early_adopters_number + early_majority_number):(
                                         innovators_number + early_adopters_number + early_majority_number + late_majority_number)]
            laggards_list = prospects[
                            (innovators_number + early_adopters_number + early_majority_number + late_majority_number):]

            for prospect in innovators_list:
                prospect.adopter_group = "Innovator"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.85, scale=np.sqrt(0.05))
            for prospect in early_adopters_list:
                prospect.adopter_group = "Early Adopter"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.7, scale=np.sqrt(0.05))
            for prospect in early_majority_list:
                prospect.adopter_group = "Early Majority"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.5, scale=np.sqrt(0.05))
            for prospect in late_majority_list:
                prospect.adopter_group = "Late Majority"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.3, scale=np.sqrt(0.05))
            for prospect in laggards_list:
                prospect.adopter_group = "Laggard"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.15, scale=np.sqrt(0.05))

        elif scenario == "Fragmented":
            innovators_number = int(0.005 * len(prospects))
            early_adopters_number = int(0.075 * len(prospects))
            early_majority_number = int(0.34 * len(prospects))
            late_majority_number = int(0.34 * len(prospects))
            laggards_number = int(0.24 * len(prospects))

            innovators_list = prospects[:innovators_number]
            early_adopters_list = prospects[innovators_number:(innovators_number + early_adopters_number)]
            early_majority_list = prospects[(innovators_number + early_adopters_number):(
                    innovators_number + early_adopters_number + early_majority_number)]
            late_majority_list = prospects[
                                 (innovators_number + early_adopters_number + early_majority_number):(
                                         innovators_number + early_adopters_number + early_majority_number + late_majority_number)]
            laggards_list = prospects[
                            (innovators_number + early_adopters_number + early_majority_number + late_majority_number):]

            for prospect in innovators_list:
                prospect.adopter_group = "Innovator"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.85, scale=np.sqrt(0.05))
            for prospect in early_adopters_list:
                prospect.adopter_group = "Early Adopter"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.7, scale=np.sqrt(0.05))
            for prospect in early_majority_list:
                prospect.adopter_group = "Early Majority"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.5, scale=np.sqrt(0.05))
            for prospect in late_majority_list:
                prospect.adopter_group = "Late Majority"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.3, scale=np.sqrt(0.05))
            for prospect in laggards_list:
                prospect.adopter_group = "Laggard"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.15, scale=np.sqrt(0.05))

        elif scenario == "Unified":
            innovators_number = int(0.065 * len(prospects))
            early_adopters_number = int(0.175 * len(prospects))
            early_majority_number = int(0.34 * len(prospects))
            late_majority_number = int(0.34 * len(prospects))
            laggards_number = int(0.08 * len(prospects))

            innovators_list = prospects[:innovators_number]
            early_adopters_list = prospects[innovators_number:(innovators_number + early_adopters_number)]
            early_majority_list = prospects[(innovators_number + early_adopters_number):(
                    innovators_number + early_adopters_number + early_majority_number)]
            late_majority_list = prospects[
                                 (innovators_number + early_adopters_number + early_majority_number):(
                                         innovators_number + early_adopters_number + early_majority_number + late_majority_number)]
            laggards_list = prospects[
                            (innovators_number + early_adopters_number + early_majority_number + late_majority_number):]

            for prospect in innovators_list:
                prospect.adopter_group = "Innovator"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.85, scale=np.sqrt(0.05))
            for prospect in early_adopters_list:
                prospect.adopter_group = "Early Adopter"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.7, scale=np.sqrt(0.05))
            for prospect in early_majority_list:
                prospect.adopter_group = "Early Majority"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.5, scale=np.sqrt(0.05))
            for prospect in late_majority_list:
                prospect.adopter_group = "Late Majority"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.3, scale=np.sqrt(0.05))
            for prospect in laggards_list:
                prospect.adopter_group = "Laggard"
                prospect.receptivity_towards_innovation = np.random.normal(loc=0.15, scale=np.sqrt(0.05))

    def check_friends_and_join(self):
        prospects = [agent for agent in self.schedule.agents if isinstance(agent, Prospect)]
        for prospect in prospects:
            if prospect.status == "New Member":
                pass
            else:
                if len(prospect.friends) != 0:
                    # Count the number of friends who have joined the community
                    num_neighbors_joined = sum(1 for friend in prospect.friends if friend.status == "New Member")

                    # Define the percentage (threshold) of friends needed based on world narrative
                    if self.scenario == "Familiar":
                        # Define the percentage (threshold) of friends needed based on adopter group
                        if prospect.adopter_group == "Innovator":
                            percent_neighbors_needed = np.random.normal(loc=0.2, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Early Adopter":
                            percent_neighbors_needed = np.random.normal(loc=0.4, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Early Majority":
                            percent_neighbors_needed = np.random.normal(loc=0.6, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Late Majority":
                            percent_neighbors_needed = np.random.normal(loc=0.75, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Laggard":
                            percent_neighbors_needed = np.random.normal(loc=0.9, scale=np.sqrt(0.05))

                        # Calculate the required number of friends needed for the prospect to consider joining
                        num_required_neighbors = len(prospect.friends) * percent_neighbors_needed

                        # If more than the required number of friends have joined, the prospect has a 50% chance of joining
                        if num_neighbors_joined > num_required_neighbors:
                            if np.random.rand() < 0.5:  # 50% chance of joining
                                prospect.status = "New Member"

                    elif self.scenario == "Unified":
                        # Define the percentage (threshold) of friends needed based on adopter group
                        if prospect.adopter_group == "Innovator":
                            percent_neighbors_needed = np.random.normal(loc=0.2, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Early Adopter":
                            percent_neighbors_needed = np.random.normal(loc=0.4, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Early Majority":
                            percent_neighbors_needed = np.random.normal(loc=0.6, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Late Majority":
                            percent_neighbors_needed = np.random.normal(loc=0.75, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Laggard":
                            percent_neighbors_needed = np.random.normal(loc=0.9, scale=np.sqrt(0.05))

                        # Calculate the required number of friends needed for the prospect to consider joining
                        num_required_neighbors = len(prospect.friends) * percent_neighbors_needed

                        # If more than the required number of friends have joined, the prospect has a 50% chance of joining
                        if num_neighbors_joined > num_required_neighbors:
                            if np.random.rand() < 0.6:  # 60% chance of joining
                                prospect.status = "New Member"

                    elif self.scenario == "Fragmented":
                        # Define the percentage (threshold) of friends needed based on adopter group
                        if prospect.adopter_group == "Innovator":
                            percent_neighbors_needed = np.random.normal(loc=0.2, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Early Adopter":
                            percent_neighbors_needed = np.random.normal(loc=0.4, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Early Majority":
                            percent_neighbors_needed = np.random.normal(loc=0.6, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Late Majority":
                            percent_neighbors_needed = np.random.normal(loc=0.75, scale=np.sqrt(0.05))
                        elif prospect.adopter_group == "Laggard":
                            percent_neighbors_needed = np.random.normal(loc=0.9, scale=np.sqrt(0.05))

                        # Calculate the required number of friends needed for the prospect to consider joining
                        num_required_neighbors = len(prospect.friends) * percent_neighbors_needed

                        # If more than the required number of friends have joined, the prospect has a 50% chance of joining
                        if num_neighbors_joined > num_required_neighbors:
                            if np.random.rand() < 0.4:  # 40% chance of joining
                                prospect.status = "New Member"
                else:
                    pass

    def prospects_find_peers(self):
        prospects = [agent for agent in self.schedule.agents if isinstance(agent, Prospect)]

        for prospect in prospects:
            potential_friends = [a for a in prospects]
            prospect.friends = random.sample(potential_friends, prospect.number_of_friends)

    def extract_typology_dataframe(self):
        # A list of dictionaries containing agent index and typology
        typology_data = [{'Index': agent.unique_id, 'Typology': agent.typology} for agent in self.schedule.agents if
                         isinstance(agent, Member)]

        # Convert the list of dictionaries to a DataFrame
        typology_df = pd.DataFrame(typology_data)

        return typology_df

    def plot_agent_type_histogram(self):
        # Get the typologies of all members
        member_typologies = [agent.typology for agent in self.schedule.agents if isinstance(agent, Member)]

        # Count the occurrences of each agent type
        type_counts = {typology: member_typologies.count(typology) for typology in set(member_typologies)}

        # Generate different colors for each bar using the 'viridis' color map
        colors = cm.viridis(np.linspace(0, 1, len(type_counts)))

        # Plot the histogram with different colors for each bar
        plt.bar(type_counts.keys(), type_counts.values(), color=colors)
        plt.xlabel('Agent Type')
        plt.ylabel('Frequency')
        plt.title('Histogram of Member Agent Types')
        plt.show()

    def collect_new_members(self):
        return self.new_members_count

    def collect_growth_percentage(self):
        growth_percentage = (self.new_members_count / self.num_members) * 100
        self.percentage_list.append(growth_percentage)
        return growth_percentage

    def check_for_new_members(self):
        for prospect in self.schedule.agents:
            if isinstance(prospect, Prospect) and prospect.status == "New Member":
                # Increment new_members_count if prospect is a Prospect with status "New Member"
                self.new_members_count += 1
                # Cap new_members_count at num_prospects
                if self.new_members_count > self.num_prospects:
                    self.new_members_count = self.num_prospects
        self.new_members_count_list.append(self.new_members_count)
        return self.new_members_count, self.new_members_count_list

    def contact_and_influence(self):
        # Current members search for prospective members and have a chance at convincing them
        members = [member for member in self.schedule.agents if isinstance(member, Member)]
        for member in members:
            prospective_agents = [neighbor for neighbor in
                                  self.grid.get_neighbors(member.pos, moore=True, include_center=False,
                                                          radius=member.reach) if isinstance(neighbor, Prospect)]
            for prospective_agent in prospective_agents:
                if prospective_agent.receptivity_towards_innovation is not None:
                    if random.random() < abs(prospective_agent.receptivity_towards_innovation * member.convincing_prowess):
                        prospective_agent.status = "New Member"
                        self.num_attempts += 1
                    else:
                        self.num_attempts += 1

    def plot_new_additions(self):
        data = self.datacollector.get_model_vars_dataframe()
        plt.plot(data.index, data['New Members'])
        plt.xlabel('Steps')
        plt.ylabel('Number of New Members')
        plt.title('Number of New Members Joined Over Time')
        plt.show()

    def members_step(self):
        members = [member for member in self.schedule.agents if isinstance(member, Member)]
        for member in members:
            member.step()

    def prospects_step(self):
        prospects = [agent for agent in self.schedule.agents if isinstance(agent, Prospect)]
        for prospect in prospects:
            prospect.step()

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        self.contact_and_influence()
        self.check_friends_and_join()
        self.check_for_new_members()

    def run_model(self):
        cumulative_members = []
        step_number = 0
        while self.new_members_count < self.num_prospects:
            self.step()
            cumulative_members.append(self.new_members_count)
            step_number += 1
        return cumulative_members
