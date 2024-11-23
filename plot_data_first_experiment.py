import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data Setup
# -----------------------------

# The data that is shown here, is based on our manual analysis of the results of running the experiments.

# Original Experiment Data
assertivness_needed_moral_disputes = [1, 3, 1, 3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 1, 2, 1, 1, 3, 1, 2, 2, 2, 1, 1, 3, 2, 1, 3, 2, 3, 3, 3, 2, 2, 1, 1, 2, 3, 2, 3, 2, 2, 2, 1, 3, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 3, 1, 1, 3, 2, 1, 1, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 2, 1, 1, 1, 3, 1, 1, 1, 2, 3, 2, 3, 3, 2, 1, 3, 2]
assertivness_needed_prehistory = [2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2]
assertivness_needed_high_school_psychology = [3, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 5, 2, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 2, 3, 3, 2, 3, 5, 3, 2, 3, 3, 2, 2, 3, 2, 5, 2, 5, 2, 2, 5, 2, 2, 2, 3, 2, 3, 5, 5, 2, 2, 5, 5, 2, 2, 5, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 2, 2]
assertivness_needed_high_school_macroeconomics = [2, 3, 3, 3, 2, 1, 2, 1, 3, 1, 3, 3, 2, 2, 4, 2, 2, 3, 3, 3, 4, 4, 2, 2, 2, 2, 4, 4, 4, 3, 3, 3, 4, 2, 3, 2, 4, 2, 4, 4, 4, 2, 2, 3, 2, 2, 1, 2, 2, 4, 1, 3, 4, 1, 2, 2, 4, 3, 3, 2, 3, 3, 4, 2, 2, 4, 4, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 1, 1, 2, 1, 3, 3, 2, 2, 3, 3, 3, 2, 4, 2, 2, 4, 4, 3, 1, 2]
assertivness_needed_moral_scenarios = [3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 2, 2, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3]
assertivness_needed_professional_psychology = [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 4, 2, 4, 2, 2, 1, 3, 3, 3, 2, 1, 3, 3, 1, 4, 2, 3, 1, 4, 1, 1, 3, 3, 2, 3, 2, 2, 4, 3, 2, 2, 3, 3, 2, 2, 2, 2, 4, 2, 2, 3, 3, 2, 3, 2, 2, 3, 3, 4, 2, 2, 3, 3, 3, 1, 3, 2, 4, 1, 4, 2, 4, 3, 2, 2, 3, 2, 3, 2, 2, 3]
assertivness_needed_elementary_mathematics = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
assertivness_needed_miscellaneous = [4, 4, 1, 4, 4, 4, 4, 4, 4, 2, 1, 4, 4, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 2, 2, 4, 4, 1, 4, 4, 1, 4, 4, 4, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4, 4, 2, 4, 1, 4, 2, 1, 4, 4, 1, 4, 2, 4, 1, 4, 4, 2, 4, 1, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 1, 4, 4, 4, 4, 1, 4, 2, 4, 4, 4, 4, 4, 4, 1, 2, 1, 4, 4, 4, 1, 4, 1, 2, 1, 4]
assertivness_needed_philosophy = [2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 3, 2, 3, 3, 2, 3, 1, 3, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 3, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 2, 1, 1, 2, 3, 2, 3, 3, 3, 2, 2, 3, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2]
assertivness_needed_professional_law = [2, 1, 1, 2, 2, 2, 2, 1, 3, 4, 1, 1, 2, 2, 1, 2, 2, 4, 2, 2, 4, 3, 4, 2, 1, 2, 2, 1, 3, 4, 2, 4, 3, 2, 4, 1, 2, 4, 2, 2, 1, 1, 4, 2, 1, 2, 2, 1, 2, 2, 4, 2, 2, 2, 2, 2, 2, 1, 4, 3, 2, 1, 4, 1, 4, 2, 2, 2, 3, 1, 2, 1, 4, 2, 2, 2, 4, 3, 2, 2, 2, 4, 1, 3, 4, 2, 3, 2, 3, 4, 2, 2, 4, 2, 2, 2, 2, 1, 4, 2]

# Repeated Experiment Data
assertivness2_needed_moral_disputes = [1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2]
assertivness2_needed_prehistory = [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1]
assertivness2_needed_high_school_psychology = [2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 3, 2, 1, 2, 3, 1, 3, 2, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 2, 1, 2, 2, 3, 1, 2, 2, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1]
assertivness2_needed_high_school_macroeconomics = [3, 1, 3, 2, 3, 1, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 3, 2, 1, 2, 3, 3, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 1, 3, 3, 3, 1, 3, 3, 1, 2, 1, 2, 1, 2, 2, 3, 1, 2, 2, 1, 2, 1, 3, 2, 2]
assertivness2_needed_moral_scenarios = [3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 3, 2, 2, 2, 3, 3, 3, 2, 1, 2, 2, 1, 3, 3, 2, 1, 1, 2, 3, 2, 2, 3, 2, 1, 2, 2, 3, 2, 2, 3, 3, 3, 2, 1, 3, 2, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 1, 1, 2, 3, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 1, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2]
assertivness2_needed_professional_psychology = [2, 3, 1, 2, 1, 3, 2, 2, 1, 2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 1, 2, 2, 2, 1, 3, 3, 2, 1, 2, 2, 3, 3, 2, 2, 1, 3, 2, 2, 3, 2, 1, 2, 1, 2, 2, 1, 3, 2, 3, 3, 2, 1, 2, 2, 2, 1, 1, 3, 3, 2, 2, 3, 2, 3, 3, 3, 3, 1, 2, 1, 3, 3, 2, 3, 2, 1, 3, 2, 2, 1, 2, 2, 2, 3, 3, 2, 2, 3, 3, 1, 3, 2, 2, 1, 2, 1, 2, 2, 2, 2]
assertivness2_needed_elementary_mathematics = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
assertivness2_needed_miscellaneous = [1, 2, 2, 2, 1, 3, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2]
assertivness2_needed_philosophy = [1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2]
assertivness2_needed_professional_law = [2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]


# Data single experiment - 5's Ignored
assertivness3_needed_moral_disputes = [1,2,2,3,3,3,1,3,1,4,4,1,3,2,1,4,3,4,3,3,4,4,3,3,3,4,3,3,3,4,3,2,3,3,3,3,3,4,3,4,4,1,3,4,1,4,3,3,3,4,2,4,3,3,3,4,4,3,1,3,1,3,3,4,2,2,2,1,2,3,3,3,3,3,2,4,4,3,3,3]
assertivness3_needed_prehistory = [1,3,3,3,3,3,3,2,3,2,1,3,2,3,3,2,2,3,3,3,2,3,3,2,3,2,1,4,3,3,3,2,3,3,3,2,3,3,3,4,2,3,2,1,4,2,2,3,3,3,3,3,3,3,3,4,1,3,2,1,3,3,2,1,1,2,3,3,2,3,3,4,3,2,1,3,4,3,3,3,4,3,1,4,4,3,3,3,4,2]
assertivness3_needed_high_school_psychology = [4,4,4,4,4,4,4,4,4,4]
assertivness3_needed_high_school_macroeconomics = [1,3,3,1,3,2,2,3,3,2,3,3,2,3,1,1,3,3,2,3,3,2,1,3,3,1,1,3,2,4,1,3,3,3,1,4,4,3,2,3,3,3,4,4,3,2,3,4,3,4,4,4,3,2,3,3,3,4,1,3]
assertivness3_needed_moral_scenarios = [3,2,3,2,1,2,2,2,2,1,2,2,2,3,2,1,2,1,1,1,2,2,2,3,2,2,1,2,2,2,2,1,3,3,2,1,2,2,2,2,2,2,2,3,2,2,3,2,2,3,1,2,2,2,1,1,3,3,2,3,3,3,2,1,2,2,2,1,1,2,2,1,2,2,1,2,2,2,2,3,2,2,2,2,2,2,2,2,3,2,2,1,3,2,3,2,3,3,1,1]
assertivness3_needed_professional_psychology = [2,2,2,3,3,2,4,2,3,2,2,4,2,2,2,3,2,3,2,2,4,2,3,2,3,2,4,3,4,3,4,4,2,4,2,3,3,2,2,2,3,2,2,2,2,4,2,2,3,3,3,2,2,2,2,3,2,2,2,3,2,2,4,2,2,3,3,2,2,3]
assertivness3_needed_elementary_mathematics = [3,3,3,3,3,3,3,3,3,3]
assertivness3_needed_miscellaneous = [2,2,4,4,2,3,3,4,4,4,4,2,3,2,4,4,4,2,3,3,3,2,2,3,2,4,3,2,3,3]
assertivness3_needed_philosophy = [3,3,1,1,3,3,3,3,3,3,2,3,2,2,3,3,3,1,3,2,1,2,2,1,2,3,2,3,2,2,2,3,3,2,3,3,3,3,3,2,2,2,2,1,3,1,2,2,1,3,2,3,3,3,1,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3]
assertivness3_needed_professional_law = [2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,1,2,2,2,1,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,2,1,2,2,2,2,2]


#Subjects List
subjects = [
    "Moral Disputes",
    "Prehistory",
    "HS Psychology",
    "HS Macroeconomics",
    "Moral Scenarios",
    "Pro Psychology",
    "Elementary Mathematics",
    "Miscellaneous",
    "Philosophy",
    "Pro Law"
]

# -----------------------------
# Calculate Average Assertiveness
# -----------------------------

# Function to calculate average assertiveness, handling empty lists
def calculate_average(data_list):
    return [np.mean(subject) if len(subject) > 0 else 0 for subject in data_list]

# Organize data into lists
original_data = [
    assertivness_needed_moral_disputes,
    assertivness_needed_prehistory,
    assertivness_needed_high_school_psychology,
    assertivness_needed_high_school_macroeconomics,
    assertivness_needed_moral_scenarios,
    assertivness_needed_professional_psychology,
    assertivness_needed_elementary_mathematics,
    assertivness_needed_miscellaneous,
    assertivness_needed_philosophy,
    assertivness_needed_professional_law
]

repeated_data = [
    assertivness2_needed_moral_disputes,
    assertivness2_needed_prehistory,
    assertivness2_needed_high_school_psychology,
    assertivness2_needed_high_school_macroeconomics,
    assertivness2_needed_moral_scenarios,
    assertivness2_needed_professional_psychology,
    assertivness2_needed_elementary_mathematics,
    assertivness2_needed_miscellaneous,
    assertivness2_needed_philosophy,
    assertivness2_needed_professional_law
]

single_data = [
    assertivness3_needed_moral_disputes,
    assertivness3_needed_prehistory,
    assertivness3_needed_high_school_psychology,
    assertivness3_needed_high_school_macroeconomics,
    assertivness3_needed_moral_scenarios,
    assertivness3_needed_professional_psychology,
    assertivness3_needed_elementary_mathematics,
    assertivness3_needed_miscellaneous,
    assertivness3_needed_philosophy,
    assertivness3_needed_professional_law
]

# Calculate averages
average_assertivness_original = calculate_average(original_data)
average_assertivness_repeated = calculate_average(repeated_data)
average_assertivness_single = calculate_average(single_data)

# -----------------------------
# Plot Average Assertiveness Comparisons
# -----------------------------

# Function to plot average assertiveness
def plot_average_comparison(subjects, avg1, avg2, label1, label2, title, colors):
    barWidth = 0.35
    r1 = np.arange(len(subjects))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(14, 7))
    plt.bar(r1, avg1, color=colors[0], width=barWidth, edgecolor='grey', label=label1)
    plt.bar(r2, avg2, color=colors[1], width=barWidth, edgecolor='grey', label=label2)

    plt.xlabel("Subjects", fontweight='bold', fontsize=12)
    plt.ylabel("Average Assertiveness", fontweight='bold', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks([r + barWidth/2 for r in range(len(subjects))], subjects, rotation=45, fontsize=10)
    plt.ylim(0, 5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot Original vs Single
plot_average_comparison(
    subjects,
    average_assertivness_original,
    average_assertivness_single,
    "Original Experiment",
    "Single Experiment",
    "Average Assertiveness: Original vs Single",
    ['#1f77b4', '#2ca02c']  # Blue and Green
)

# Plot Single vs Repeated
plot_average_comparison(
    subjects,
    average_assertivness_single,
    average_assertivness_repeated,
    "Single Experiment",
    "Repeated Experiment",
    "Average Assertiveness: Single vs Repeated",
    ['#2ca02c', '#ff7f0e']  # Green and Orange
)

# -----------------------------
# Calculate Percentage of Changed Answers
# -----------------------------

# Function to calculate percentage of changes
def calculate_percentage_changes(original, comparison):
    changes = []
    assert_levels = []
    for orig_subject, comp_subject in zip(original, comparison):
        min_length = min(len(orig_subject), len(comp_subject))
        for j in range(min_length):
            changes.append(1 if orig_subject[j] != comp_subject[j] else 0)
            assert_levels.append(comp_subject[j])
    levels = [1, 2, 3, 4]
    percent_changes = []
    for level in levels:
        indices = [i for i, x in enumerate(assert_levels) if x == level]
        total_at_level = len(indices)
        if total_at_level > 0:
            changes_at_level = sum([changes[i] for i in indices])
            percent_change = changes_at_level / total_at_level
        else:
            percent_change = 0
        percent_changes.append(percent_change)
    return percent_changes

# Calculate changes for Original vs Single and Original vs Repeated
percent_changes_original_vs_single = calculate_percentage_changes(original_data, single_data)
percent_changes_original_vs_repeated = calculate_percentage_changes(original_data, repeated_data)

# -----------------------------
# Plot Percentage of Changed Answers Comparisons
# -----------------------------

# Function to plot percentage changes
def plot_percentage_comparison(levels, perc1, perc2, label1, label2, title, colors):
    barWidth = 0.35
    r1 = np.arange(len(levels))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(10, 6))
    plt.bar(r1, perc1, color=colors[0], width=barWidth, edgecolor='grey', label=label1)
    plt.bar(r2, perc2, color=colors[1], width=barWidth, edgecolor='grey', label=label2)

    plt.xlabel("Assertiveness Level", fontweight='bold', fontsize=12)
    plt.ylabel("Percentage of Changed Answers", fontweight='bold', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks([r + barWidth/2 for r in range(len(levels))], levels, fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot Original vs Single
plot_percentage_comparison(
    levels=[1, 2, 3, 4],
    perc1=percent_changes_original_vs_single,
    perc2=[],  # No second comparison for this plot
    label1="Original vs Single",
    label2="",
    title="Percentage of Changed Answers: Original vs Single",
    colors=['#2ca02c', '#d62728']  # Green and Red (Red unused)
)

# Plot Original vs Repeated
plot_percentage_comparison(
    levels=[1, 2, 3, 4],
    perc1=percent_changes_original_vs_repeated,
    perc2=[],  # No second comparison for this plot
    label1="Original vs Repeated",
    label2="",
    title="Percentage of Changed Answers: Original vs Repeated",
    colors=['#ff7f0e', '#d62728']  # Orange and Red (Red unused)
)

# -----------------------------
# Optional: Annotate Bars with Percentage Labels
# -----------------------------

# Function to annotate bars with percentage labels
def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

# Redefine plot_percentage_comparison to include annotations
def plot_percentage_comparison_with_annotations(levels, perc, label, title, color):
    barWidth = 0.5
    r = np.arange(len(levels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(r, perc, color=color, width=barWidth, edgecolor='grey')

    ax.set_xlabel("Assertiveness Level", fontweight='bold', fontsize=12)
    ax.set_ylabel("Percentage of Changed Answers", fontweight='bold', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(r)
    ax.set_xticklabels(levels, fontsize=12)
    ax.set_ylim(0, 1)

    # Annotate bars
    annotate_bars(ax, bars)

    plt.tight_layout()
    plt.show()

# Plot Original vs Single with Annotations
plot_percentage_comparison_with_annotations(
    levels=[1, 2, 3, 4],
    perc=percent_changes_original_vs_single,
    label="Original vs Single",
    title="Percentage of Changed Answers: Original vs Single",
    color='#2ca02c'  # Green
)

# Plot Original vs Repeated with Annotations
plot_percentage_comparison_with_annotations(
    levels=[1, 2, 3, 4],
    perc=percent_changes_original_vs_repeated,
    label="Original vs Repeated",
    title="Percentage of Changed Answers: Original vs Repeated",
    color='#ff7f0e'  # Orange
)
