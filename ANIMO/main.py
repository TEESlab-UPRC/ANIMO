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

import pandas as pd
from community import EnergyCommunityModel

# Empty DataFrame to store the results
df_simulation_results = pd.DataFrame()

"""Specify to configure a batch run"""
batch_runs_number = 10  # You can increase this for multiple runs

"""Specify number of instances from each class"""
num_members = 100
num_prospects = 100

"""Specify scenario configuration"""

"""Possible choices for scenario parameter:
-> "Fragmented"
-> "Familiar"
-> "Unified"  """

#  Current configuration:
scenario = 'Fragmented'

for i in range(batch_runs_number):
    # Instantiate the model
    model = EnergyCommunityModel(num_members=num_members, num_prospects=num_prospects, height=500,
                                 width=500, scenario=scenario)

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
