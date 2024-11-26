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


import pandas as pd
from community import EnergyCommunityModel

# Empty DataFrame to store the results
df_simulation_results = pd.DataFrame()

"""Import input data"""
file_path = 'input_data.xlsx'
data = pd.read_excel(file_path)

"""Specify height and width of the simulation space"""
height = data.loc[data['Unnamed: 0'] == 'height', 'value'].iloc[0]
width = data.loc[data['Unnamed: 0'] == 'width', 'value'].iloc[0]

"""Specify to configure a batch run"""
batch_runs_number = 10  # You can increase this for multiple runs

"""Specify number of instances from each class"""
num_members = data.loc[data['Unnamed: 0'] == 'num_members', 'value'].iloc[0]
num_prospects = data.loc[data['Unnamed: 0'] == 'num_prospects', 'value'].iloc[0]

"""Specify scenario configuration"""
scenario = data.loc[data['Unnamed: 0'] == 'scenario', 'value'].iloc[0]

for i in range(batch_runs_number):
    # Instantiate the model
    model = EnergyCommunityModel(num_members=num_members, num_prospects=num_prospects, height=height,
                                 width=width, scenario=scenario)

    # Run the model and get the cumulative members list
    cumulative_members = model.run_model()

    # Execute the following to get histogram of members types
    # model.plot_agent_type_histogram()

    # Execute the following to plot the increasing cumulative number of members
    model.plot_new_additions()

    # Create a step column
    step_numbers = list(range(1, len(cumulative_members) + 1))

    # Add step numbers and results to the DataFrame
    df_simulation_results[f'Step Number Iteration #{i + 1}'] = pd.Series(step_numbers)
    df_simulation_results[f'New Members Iteration #{i + 1}'] = pd.Series(cumulative_members)

# Exporting DataFrame to a CSV file
df_simulation_results.to_csv('simulation_results.csv', index=False)

print("Simulation results saved to 'simulation_results.csv'")
