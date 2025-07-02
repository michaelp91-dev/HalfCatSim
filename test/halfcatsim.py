import pandas as pd
import math
import csv
import string
import numpy as np
from scipy.stats import linregress

def INTERCEPT(
    dataframe: pd.DataFrame,
    y_column_name: str,
    x_column_name: str,
    start_row: int,
    end_row: int
  ) -> float:
    if y_column_name not in dataframe.columns:
        raise ValueError(f"Y-column '{y_column_name}' not found in the DataFrame.")
    if x_column_name not in dataframe.columns:
        raise ValueError(f"X-column '{x_column_name}' not found in the DataFrame.")

    num_rows = len(dataframe)
    if not (0 <= start_row <= end_row < num_rows):
        raise ValueError(f"Invalid row range: start_row={start_row}, end_row={end_row}. "
                         f"DataFrame has {num_rows} rows (0-indexed up to {num_rows - 1}).")

    selected_df_range = dataframe.iloc[start_row : end_row + 1]

    y_series = selected_df_range[y_column_name]
    x_series = selected_df_range[x_column_name]

    combined_data = pd.DataFrame({
        'x': x_series,
        'y': y_series
    }).dropna()

    if combined_data.empty:
        raise ValueError("No valid (non-NaN) data points found in the specified columns and row range for regression.")

    try:
        x_values = pd.to_numeric(combined_data['x'], errors='coerce').dropna().values
        y_values = pd.to_numeric(combined_data['y'], errors='coerce').dropna().values
    except Exception as e:
        raise ValueError(f"Could not convert data in specified columns to numeric: {e}")

    if len(x_values) != len(y_values):
        raise ValueError("After numeric conversion and NaN removal, X and Y data points do not align for regression.")

    if len(x_values) < 2:
        raise ValueError("Not enough valid numeric data points (need at least 2) for linear regression in the specified range.")

    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

    return intercept

def _col_to_index(col):
    if isinstance(col, int):
        if col < 1:
            raise ValueError(f"Column number must be 1 or greater, got {col}.")
        return col - 1
    elif isinstance(col, str):
        col = col.upper()
        index = 0
        for char in col:
            if not 'A' <= char <= 'Z':
                raise ValueError(f"Invalid column letter: '{col}'. Must be A-Z.")
            index = index * 26 + (ord(char) - ord('A') + 1)
        return index - 1
    else:
        raise TypeError(f"Column must be an integer or a string letter, got {type(col)}.")

def slice_data(full_table_data, start_row, end_row, start_col, end_col):
    if not isinstance(full_table_data, list) or not full_table_data:
        raise ValueError("full_table_data must be a non-empty list of lists.")
    if not all(isinstance(row, list) for row in full_table_data):
        raise ValueError("full_table_data must contain only lists (rows).")

    if not (isinstance(start_row, int) and start_row >= 1 and
            isinstance(end_row, int) and end_row >= start_row):
        raise ValueError("start_row and end_row must be integers, 1 or greater, and start_row <= end_row.")

    python_start_row = start_row - 1
    python_end_row = end_row

    python_start_col = _col_to_index(start_col)
    python_end_col = _col_to_index(end_col) + 1

    if python_start_col >= python_end_col:
        raise ValueError(f"start_col ('{start_col}') must be less than end_col ('{end_col}'). This indicates a backward range in Excel, which needs to be reordered (min to max column).")


    sliced_data = []
    for r_idx in range(python_start_row, min(python_end_row, len(full_table_data))):
        row = full_table_data[r_idx]
        if not row:
            sliced_data.append([])
            continue

        if len(row) > python_start_col:
            sliced_row = row[python_start_col:min(python_end_col, len(row))]
            sliced_data.append(sliced_row)
        else:
            sliced_data.append([])

    return sliced_data

def load_csv(file_path):
    table_data = []
    try:
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                processed_row = []
                for i, item in enumerate(row):
                    if i == 0:
                        try:
                            processed_row.append(float(item))
                        except ValueError:
                            processed_row.append(item)
                    else:
                        processed_row.append(item)
                table_data.append(processed_row)
        return table_data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        return None

_CONVERSION_MAP = {
    "length": {
        "m": 1.0, "meter": 1.0, "meters": 1.0, "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144, "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    },
    "mass": {
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "g": 0.001, "gram": 0.001, "grams": 0.001,
        "lbm": 0.45359237, "lb": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
        "ozm": 0.028349523125, "oz": 0.028349523125, "ounce": 0.028349523125, "ounces": 0.028349523125,
    },
    "time": {
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "min": 60.0, "minute": 60.0, "minutes": 60.0,
        "hr": 3600.0, "hour": 3600.0, "hours": 3600.0, "day": 86400.0, "days": 86400.0,
    },
    "volume": {
        "l": 1.0, "liter": 1.0, "liters": 1.0, "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
        "gal": 3.785411784, "gallon": 3.785411784, "gallons": 3.785411784, "qt": 0.946352946,
        "quart": 0.946352946, "quarts": 0.946352946, "pt": 0.473176473, "pint": 0.473176473, "pints": 0.473176473,
        "cup": 0.2365882365, "cups": 0.2365882365, "foz": 0.0295735295625, "fluid_ounce": 0.0295735295625, "fluid_ounces": 0.0295735295625,
        "m3": 1000.0, "cubic_meter": 1000.0, "cubic_meters": 1000.0, "ft3": 28.316846592, "cubic_foot": 28.316846592,
        "cubic_feet": 28.316846592, "in3": 0.016387064, "cubic_inch": 0.016387064, "cubic_inches": 0.016387064,
    },
    "temperature": {
        "c": "celsius", "celsius": "celsius", "f": "fahrenheit", "fahrenheit": "fahrenheit",
        "k": "kelvin", "kelvin": "kelvin",
    },
    "pressure": {
        "pa": 1.0, "pascal": 1.0, "pascals": 1.0,
        "kpa": 1000.0, "kilopascal": 1000.0, "kilopascals": 1000.0,
        "mpa": 1000000.0, "megapascal": 1000000.0, "megapascals": 1000000.0,
        "psi": 6894.757293168, "pound_per_square_inch": 6894.757293168,
        "bar": 100000.0,
        "atm": 101325.0, "atmosphere": 101325.0, "atmospheres": 101325.0,
        "mmhg": 133.322387415, "millimeter_of_mercury": 133.322387415, "torr": 133.322387415,
        "inhg": 3386.388666667, "inch_of_mercury": 3386.388666667,
    },
    "area": {
        "m2": 1.0, "m^2": 1.0, "sqm": 1.0, "square_meter": 1.0, "square_meters": 1.0,
        "cm2": 0.0001, "cm^2": 0.0001, "sqcm": 0.0001, "square_centimeter": 0.0001, "square_centimeters": 0.0001,
        "mm2": 0.000001, "mm^2": 0.000001, "sqmm": 0.000001, "square_millimeter": 0.000001, "square_millimeters": 0.000001,
        "in2": 0.00064516, "in^2": 0.00064516, "sqin": 0.00064516, "square_inch": 0.00064516, "square_inches": 0.00064516,
        "ft2": 0.09290304, "ft^2": 0.09290304, "sqft": 0.09290304, "square_foot": 0.09290304, "square_feet": 0.09290304,
        "yd2": 0.83612736, "yd^2": 0.83612736, "sqyd": 0.83612736, "square_yard": 0.83612736, "square_yards": 0.83612736,
        "mi2": 2589988.110336, "mi^2": 2589988.110336, "sqmi": 2589988.110336, "square_mile": 2589988.110336, "square_miles": 2589988.110336,
        "acre": 4046.8564224, "acres": 4046.8564224,
        "ha": 10000.0, "hectare": 10000.0, "hectares": 10000.0,
    },
    "force": {
        "n": 1.0, "newton": 1.0, "newtons": 1.0,
        "kn": 1000.0, "kilonewton": 1000.0, "kilonewtons": 1000.0,
        "lbf": 4.44822, "pound_force": 4.44822, "pound-force": 4.44822, "pounds_force": 4.44822, "pounds-force": 4.44822,
        "kgf": 9.80665, "kilogram_force": 9.80665, "kilogram-force": 9.80665, "kilopond": 9.80665, "kp": 9.80665,
    }
}

def _get_unit_category(unit_str):
    normalized_unit = unit_str.lower()
    for category, units in _CONVERSION_MAP.items():
        if normalized_unit in units:
            return category
    return None

def _convert_temperature(value, from_unit, to_unit):
    normalized_from = from_unit.lower()
    normalized_to = to_unit.lower()

    if normalized_from == "c" or normalized_from == "celsius":
        celsius_value = value
    elif normalized_from == "f" or normalized_from == "fahrenheit":
        celsius_value = (value - 32) * 5 / 9
    elif normalized_from == "k" or normalized_from == "kelvin":
        celsius_value = value - 273.15
    else:
        raise ValueError(f"Unknown temperature 'from' unit: {from_unit}")

    if normalized_to == "c" or normalized_to == "celsius":
        return celsius_value
    elif normalized_to == "f" or normalized_to == "fahrenheit":
        return (celsius_value * 9 / 5) + 32
    elif normalized_to == "k" or normalized_to == "kelvin":
        return celsius_value + 273.15
    else:
        raise ValueError(f"Unknown temperature 'to' unit: {to_unit}")


def CONVERT(number, from_unit, to_unit):
    if not isinstance(number, (int, float)):
        raise ValueError("CONVERT function: 'number' must be a numeric value.")

    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()

    from_category = _get_unit_category(from_unit_lower)
    to_category = _get_unit_category(to_unit_lower)

    if from_category is None:
        raise ValueError(f"CONVERT function: Unrecognized 'from_unit': '{from_unit}'")
    if to_category is None:
        raise ValueError(f"CONVERT function: Unrecognized 'to_unit': '{to_unit}'")

    if from_category != to_category:
        raise ValueError(f"CONVERT function: Cannot convert between incompatible unit categories ('{from_category}' and '{to_category}').")

    if from_category == "temperature":
        return _convert_temperature(number, from_unit_lower, to_unit_lower)
    else:
        try:
            from_factor = _CONVERSION_MAP[from_category][from_unit_lower]
            base_value = number * from_factor
            to_factor = _CONVERSION_MAP[to_category][to_unit_lower]
            converted_value = base_value / to_factor
            return converted_value
        except KeyError:
            raise ValueError("Internal error: Unit not found in conversion map despite category being identified.")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred during conversion: {e}")

def VLOOKUP(lookup_value, table_array, col_index_num, range_lookup=True):
    if not isinstance(table_array, (list, tuple)) or not table_array:
        raise ValueError("VLOOKUP: 'table_array' must be a non-empty list or tuple of rows.")

    if not isinstance(col_index_num, int) or col_index_num < 1:
        raise ValueError("VLOOKUP: 'col_index_num' must be an integer greater than or equal to 1.")

    target_col_index = col_index_num - 1

    if not table_array[0]:
        raise ValueError("VLOOKUP: 'table_array' contains empty rows.")
    if target_col_index >= len(table_array[0]):
        raise ValueError(f"VLOOKUP: 'col_index_num' ({col_index_num}) is out of bounds. "
                         f"Table has {len(table_array[0])} columns.")

    if range_lookup is False:
        for row in table_array:
            if not row: continue
            try:
                if isinstance(lookup_value, (int, float)):
                    first_col_value = type(lookup_value)(row[0])
                else:
                    first_col_value = row[0]
            except ValueError:
                continue

            if first_col_value == lookup_value:
                return row[target_col_index]
        return None
    else:
        last_suitable_match = None
        for row in table_array:
            if not row: continue
            try:
                first_col_value = type(lookup_value)(row[0])
            except ValueError:
                continue

            if first_col_value <= lookup_value:
                last_suitable_match = row[target_col_index]
            else:
                break
        return last_suitable_match

def MATCH(lookup_value, lookup_array, match_type=0):
    if match_type == 0:
        for i, val in enumerate(lookup_array):
            try:
                if isinstance(lookup_value, (int, float)):
                    val_converted = float(val)
                else:
                    val_converted = val
            except ValueError:
                if isinstance(lookup_value, (int, float)):
                    continue
                else:
                    val_converted = val
            if val_converted == lookup_value:
                return i + 1
        return None
    elif match_type == 1:
        last_suitable_match_index = None
        for i, val in enumerate(lookup_array):
            try:
                val_converted = float(val)
            except ValueError:
                continue
            if val_converted <= lookup_value:
                last_suitable_match_index = i + 1
            else:
                break
        return last_suitable_match_index
    elif match_type == -1:
        last_suitable_match_index = None
        for i, val in enumerate(lookup_array):
            try:
                val_converted = float(val)
            except ValueError:
                continue
            if val_converted >= lookup_value:
                last_suitable_match_index = i + 1
            else:
                break
        return last_suitable_match_index
    else:
        raise ValueError("MATCH: 'match_type' must be 0 for exact match, 1 for less than or exact match, or -1 for greater than or exact match.")

def SIGN(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def calculate_result(dataframe: pd.DataFrame, column_name: str, start_row: int, end_row: int) -> float:
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    if not (0 <= start_row <= end_row < len(dataframe)):
        raise ValueError(f"Invalid row range: start_row={start_row}, end_row={end_row}. "
                         f"DataFrame has {len(dataframe)} rows.")

    # Select the relevant slice of the Series and calculate the mean
    selected_values = dataframe.loc[start_row : end_row, column_name]

    # Check if the selected_values Series is empty (e.g., if the range was valid but no numeric data)
    if selected_values.empty:
        return 0.0 # Or raise an error, depending on desired behavior for empty ranges

    return selected_values.mean()

def run_simulation(
    n2o_filepath: str,
    fuel_filepath: str,
    formulas_filepath: str,
    calculations_filepath: str,
    isentropic_flow_filepath: str,
    timestep: float = 0.01,
    altitude: float = 2000,
    altitude_units: str = "ft",
    throat_diameter: float = 1,
    throat_diameter_units: str = "in",
    exit_diameter: float = 2,
    exit_diameter_units: str = "in",
    exit_half_angle: float = 15,
    nozzle_efficiency: float = 0.98,
    chamber_diameter: float = 2,
    chamber_diameter_units: str = "in",
    chamber_length: float = 5,
    chamber_length_units: str = "in",
    c_star_efficiency: float = 0.70,
    ox_tank_outer_diameter: float = 4,
    ox_tank_outer_diameter_units: str = "in",
    ox_wall_thickness: float = 0.125,
    ox_wall_thickness_units: str = "in",
    siphon_diameter: float = 0,
    siphon_diameter_units: str = "in",
    ox_displacement: float = 24,
    ox_displacement_units: str = "in",
    n2o_temperature: float = 50,
    n2o_temperature_units: str = "F",
    decay_constant_liquid: float = 0.7,
    decay_constant_gas: float = 0.25,
    fuel: str = "E85",
    tank_style: str = "Stacked",
    f_tank_outer_diameter: float = 4,
    f_tank_outer_diameter_units: str = "in",
    f_wall_thickness: float = 0.125,
    f_wall_thickness_units: str = "in",
    f_tank_length: float = 12,
    f_tank_length_units: str = "in",
    f_displacement: float = 9,
    f_displacement_units: str = "in",
    piston_loss: float = 15,
    piston_loss_units: str = "psi",
    ox_feed_line_diameter: float = 0.3125,
    ox_feed_line_diameter_units: str = "in",
    ox_line_roughness: float = 3.94e-06,
    ox_line_roughness_units: str = "in",
    ox_line_length: float = 12,
    ox_line_length_units: str = "in",
    ox_component_Cv: float = 1.6,
    ox_additional_loss: float = 0,
    f_feed_line_diameter: float = 0.2250,
    f_feed_line_diameter_units: str = "in",
    f_line_roughness: float = 3.94e-06,
    f_line_roughness_units: str = "in",
    f_line_length: float = 48,
    f_line_length_units: str = "in",
    f_component_Cv: float = 1.6,
    f_additional_loss: float = 0,
    regen_channel_loss: float = 0,
    ox_injector_hole_diameter: float = 0.089,
    ox_number_of_holes: int = 6,
    ox_discharge_coefficient: float = 0.4,
    f_injector_hole_diameter: float = 0.0469,
    f_number_of_holes: int = 6,
    f_discharge_coefficient: float = 0.55
):
    """
    Runs a rocket engine simulation for a blowdown feed system.

    Args:
        n2o_filepath (str): Path to the N2O data CSV file.
        fuel_filepath (str): Path to the Fuel data CSV file.
        formulas_filepath (str): Path to the Formulas data CSV file.
        calculations_filepath (str): Path to the Calculations data CSV file.
        isentropic_flow_filepath (str): Path to the Isentropic Flow data CSV file.
        timestep (float): Simulation timestep in seconds.
        altitude (float): Altitude.
        altitude_units (str): Units for altitude (e.g., "ft", "m").
        throat_diameter (float): Throat diameter.
        throat_diameter_units (str): Units for throat diameter.
        exit_diameter (float): Exit diameter.
        exit_diameter_units (str): Units for exit diameter.
        exit_half_angle (float): Exit half-angle in degrees.
        nozzle_efficiency (float): Nozzle efficiency.
        chamber_diameter (float): Chamber diameter.
        chamber_diameter_units (str): Units for chamber diameter.
        chamber_length (float): Chamber length.
        chamber_length_units (str): Units for chamber length.
        c_star_efficiency (float): C-star efficiency.
        ox_tank_outer_diameter (float): Oxidizer tank outer diameter.
        ox_tank_outer_diameter_units (str): Units for oxidizer tank outer diameter.
        ox_wall_thickness (float): Oxidizer tank wall thickness.
        ox_wall_thickness_units (str): Units for oxidizer tank wall thickness.
        siphon_diameter (float): Siphon diameter (for concentric tank).
        siphon_diameter_units (str): Units for siphon diameter.
        ox_displacement (float): Oxidizer displacement.
        ox_displacement_units (str): Units for oxidizer displacement.
        n2o_temperature (float): N2O temperature.
        n2o_temperature_units (str): Units for N2O temperature.
        decay_constant_liquid (float): Decay constant for liquid N2O.
        decay_constant_gas (float): Decay constant for gaseous N2O.
        fuel (str): Type of fuel (e.g., "E85").
        tank_style (str): Tank style ("Stacked" or "Concentric").
        f_tank_outer_diameter (float): Fuel tank outer diameter.
        f_tank_outer_diameter_units (str): Units for fuel tank outer diameter.
        f_wall_thickness (float): Fuel tank wall thickness.
        f_wall_thickness_units (str): Units for fuel tank wall thickness.
        f_tank_length (float): Fuel tank length.
        f_tank_length_units (str): Units for fuel tank length.
        f_displacement (float): Fuel displacement.
        f_displacement_units (str): Units for fuel displacement.
        piston_loss (float): Piston loss.
        piston_loss_units (str): Units for piston loss.
        ox_feed_line_diameter (float): Oxidizer feed line diameter.
        ox_feed_line_diameter_units (str): Units for oxidizer feed line diameter.
        ox_line_roughness (float): Oxidizer line roughness.
        ox_line_roughness_units (str): Units for oxidizer line roughness.
        ox_line_length (float): Oxidizer line length.
        ox_line_length_units (str): Units for oxidizer line length.
        ox_component_Cv (float): Oxidizer component Cv.
        ox_additional_loss (float): Additional oxidizer loss.
        f_feed_line_diameter (float): Fuel feed line diameter.
        f_feed_line_diameter_units (str): Units for fuel feed line diameter.
        f_line_roughness (float): Fuel line roughness.
        f_line_roughness_units (str): Units for fuel line roughness.
        f_line_length (float): Fuel line length.
        f_line_length_units (str): Units for fuel line length.
        f_component_Cv (float): Fuel component Cv.
        f_additional_loss (float): Additional fuel loss.
        regen_channel_loss (float): Regeneration channel loss.
        ox_injector_hole_diameter (float): Oxidizer injector hole diameter.
        ox_number_of_holes (int): Number of oxidizer injector holes.
        ox_discharge_coefficient (float): Oxidizer discharge coefficient.
        f_injector_hole_diameter (float): Fuel injector hole diameter.
        f_number_of_holes (int): Number of fuel injector holes.
        f_discharge_coefficient (float): Fuel discharge coefficient.

    Returns:
        dict: A dictionary containing the simulation results.
    """

    # load csv files
    n2o_data = load_csv(n2o_filepath)
    fuel_data = load_csv(fuel_filepath)
    formulas_data = load_csv(formulas_filepath)
    calculations_data = load_csv(calculations_filepath)
    isentropic_flow_data = load_csv(isentropic_flow_filepath)

    # pre calculation outputs
    max_sim_time = timestep * 2500
    ambient_pressure_units = "psi"
    ambient_pressure = CONVERT(101325 * (1 - 0.0000225577 * CONVERT(altitude, altitude_units, "m"))**5.25588, "Pa", ambient_pressure_units)
    throat_area = CONVERT(throat_diameter, throat_diameter_units, "m")**2 * math.pi / 4
    exit_area = CONVERT(exit_diameter, exit_diameter_units, "m")**2 * math.pi / 4
    expansion_ratio = exit_area / throat_area
    contraction_ratio = chamber_diameter**2 / throat_diameter**2
    chamber_volume = math.pi * CONVERT(chamber_diameter, chamber_diameter_units, "m")**2 / 4 * CONVERT(chamber_length, chamber_length_units, "m")
    L_star_units = "in"
    L_star = CONVERT(chamber_volume / throat_area, "m", L_star_units)
    ox_tank_inner_diameter = ox_tank_outer_diameter - 2 * ox_wall_thickness
    n2o_density = float(VLOOKUP(n2o_temperature, n2o_data, 3, True))
    f_tank_inner_diameter = f_tank_outer_diameter - 2 * f_wall_thickness
    fuel_density = float(VLOOKUP(fuel, slice_data(fuel_data,4,38,"E","G"), 3, False))
    oxidizer_volume = math.pi * ox_tank_inner_diameter**2 * ox_displacement / 4 - math.pi * f_tank_outer_diameter**2 * f_tank_length / 4 - math.pi * siphon_diameter**2 * (ox_displacement - f_tank_length) / 4 if tank_style == "Concentric" else math.pi * ox_tank_inner_diameter**2 * ox_displacement / 4
    if tank_style == "Concentric":
      if siphon_diameter == 0:
        if f_tank_length < (ox_displacement / 2):
          useable_volume = oxidizer_volume - math.pi * f_tank_outer_diameter**2 * f_displacement / 4
        else:
          useable_volume = math.pi * (ox_tank_inner_diameter**2 - f_tank_outer_diameter**2) * ox_displacement / 4
      else:
        useable_volume = oxidizer_volume
    else:
      useable_volume = oxidizer_volume
    total_oxidizer_mass_units = "lbm"
    total_oxidizer_mass = CONVERT(useable_volume * n2o_density / 61024, "kg", total_oxidizer_mass_units)
    oxidizer_mass_units = "lbm"
    oxidizer_mass = CONVERT(oxidizer_volume * n2o_density / 61024, "kg", oxidizer_mass_units)
    f_tank_volume = math.pi * f_tank_outer_diameter**2 * f_tank_length / 4
    fuel_volume = math.pi * f_tank_inner_diameter**2 * f_displacement / 4
    fuel_mass_units = "lbm"
    fuel_mass = CONVERT(fuel_volume * fuel_density / 61024, "kg", fuel_mass_units)
    ox_feed_line_area = CONVERT(math.pi * ox_feed_line_diameter**2 / 4, "in^2", "m^2")
    f_feed_line_area = CONVERT(math.pi * f_feed_line_diameter**2 / 4, "in^2", "m^2")
    oxidizer_CdA = CONVERT(math.pi * ox_injector_hole_diameter**2 / 4 * ox_discharge_coefficient * ox_number_of_holes, "in^2", "m^2")
    fuel_CdA = CONVERT(math.pi * f_injector_hole_diameter**2 / 4 * f_discharge_coefficient * f_number_of_holes, "in^2", "m^2")

    # Initialize the 'data' dictionary
    data = {}

    if formulas_data is None or calculations_data is None:
        print("Could not process due to file loading errors.")
        return {}
    else:
        if len(formulas_data) < 1 or len(calculations_data) < 2:
            print("Error: 'formulas.csv' needs at least one row (header) and 'Calculations.csv' needs at least two rows (header + data row).")
            return {}
        else:
            calculations_headers = calculations_data[0]
            formulas_row1 = formulas_data[0]
            calculations_row2 = calculations_data[1]

            for col_idx, header in enumerate(calculations_headers):
                key = header
                if col_idx < len(formulas_row1) and col_idx < len(calculations_row2):
                    if formulas_row1[col_idx] == 'None':
                        try:
                            data[key] = float(calculations_row2[col_idx])
                        except ValueError:
                            data[key] = calculations_row2[col_idx]
                    else:
                        data[key] = None
                else:
                    print(f"Warning: Column index {col_idx} (header '{header}') is out of bounds for row 1 in formulas.csv or row 2 in calculations.csv. Skipping.")

    calculations = []
    init_P_N2O_tank = 0 # Initialize here
    init_m_ox = 0 # Initialize here
    for i in range(2500):
      if i > 0:
        data['Time'] = None if data['F'] == 0 else data['Time'] + data['Step']
      if i == 15:
        data['Step'] = 0.01
      if i == 0:
        data['P_N2O_tank'] = float(VLOOKUP(n2o_temperature, n2o_data, 2, True))
      elif i < 4:
        if data['N2O Status'] == "Liquid":
          if data['m_ox'] > 0 and data['m_f'] > 0:
            data['P_N2O_tank'] = init_P_N2O_tank * (data['m_ox'] / init_m_ox * (1 - decay_constant_liquid) + decay_constant_liquid)
          else:
            data['P_N2O_tank'] = None
            break
        else:
          data['P_N2O_tank'] = data['P_N2O_tank'] / math.exp(decay_constant_gas * (data['Time'] - (data['Time'] - data['Step'])))
      else:
        if data['N2O Status'] == "Liquid":
          if data['m_ox'] > 0 and data['m_f'] > 0:
            data['P_N2O_tank'] = init_P_N2O_tank * (data['m_ox'] / init_m_ox * (1 - decay_constant_liquid) + decay_constant_liquid)
          else:
            data['P_N2O_tank'] = None
            break
        else:
          data['P_N2O_tank'] = data['P_N2O_tank'] / math.exp(decay_constant_gas * (data['Time'] - (data['Time'] - data['Step'])))
      data['P_N2O_tank_psi'] = CONVERT(data['P_N2O_tank'] * 1000, "Pa", "psi")
      if i < 2:
        data['P_N2O_inj'] = data['P_N2O_tank']
      else:
        data['P_N2O_inj'] = data['P_N2O_tank'] - data['dP_o_total'] / 1000
      data['P_fu_tank'] = data['P_N2O_tank'] - CONVERT(piston_loss, "psi", "Pa") / 1000
      data['P_fu_tank_psi'] = CONVERT(data['P_fu_tank'] * 1000, "Pa", "psi")
      if i < 2:
        data['P_fu_inj'] = data['P_fu_tank']
      else:
        data['P_fu_inj'] = data['P_fu_tank'] - data['dP_f_total'] / 1000
      if i >= 2:
        outer_vlookup_lookup_value = 50 if round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50 == 0 else round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50
        start_row_index = int(VLOOKUP("gam0", slice_data(fuel_data, 4, 7, 'B', 'C'), 2, False))
        start_col_excel_num = int(VLOOKUP(fuel, slice_data(fuel_data, 4, 38, 'E', 'F'), 2, False))
        main_vlookup_table_array = slice_data(fuel_data, start_row_index, start_row_index + 20, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array_sliced = slice_data(fuel_data, start_row_index, start_row_index, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array = match_lookup_array_sliced[0]
        main_vlookup_col_index = MATCH(round(data['MR'] / 0.25, 0) * 0.25, match_lookup_array, 0)
        data["gam0"] = float(VLOOKUP(outer_vlookup_lookup_value, main_vlookup_table_array, main_vlookup_col_index, True))
        outer_vlookup_lookup_value = 50 if round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50 == 0 else round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50
        start_row_index = int(VLOOKUP("MW", slice_data(fuel_data, 4, 7, 'B', 'C'), 2, False))
        start_col_excel_num = int(VLOOKUP(fuel, slice_data(fuel_data, 4, 38, 'E', 'F'), 2, False))
        main_vlookup_table_array = slice_data(fuel_data, start_row_index, start_row_index + 20, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array_sliced = slice_data(fuel_data, start_row_index, start_row_index, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array = match_lookup_array_sliced[0]
        main_vlookup_col_index = MATCH(round(data['MR'] / 0.25, 0) * 0.25, match_lookup_array, 0)
        data["MW"] = float(VLOOKUP(outer_vlookup_lookup_value, main_vlookup_table_array, main_vlookup_col_index, True))
      data['R'] = 8314 / data['MW']
      if i == 1:
        data['c*'] = 1000
      elif i >= 2 and i < 15:
        outer_vlookup_lookup_value = 50 if round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50 == 0 else round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50
        start_row_index = int(VLOOKUP("c*", slice_data(fuel_data, 4, 7, 'B', 'C'), 2, False))
        start_col_excel_num = int(VLOOKUP(fuel, slice_data(fuel_data, 4, 38, 'E', 'F'), 2, False))
        main_vlookup_table_array = slice_data(fuel_data, start_row_index, start_row_index + 20, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array_sliced = slice_data(fuel_data, start_row_index, start_row_index, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array = match_lookup_array_sliced[0]
        main_vlookup_col_index = MATCH(round(data['MR'] / 0.25, 0) * 0.25, match_lookup_array, 0)
        data["c*"] = float(VLOOKUP(outer_vlookup_lookup_value, main_vlookup_table_array, main_vlookup_col_index, True))
      elif i >= 15:
        outer_vlookup_lookup_value = 50 if round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50 == 0 else round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50
        start_row_index = int(VLOOKUP("c*", slice_data(fuel_data, 4, 7, 'B', 'C'), 2, False))
        start_col_excel_num = int(VLOOKUP(fuel, slice_data(fuel_data, 4, 38, 'E', 'F'), 2, False))
        main_vlookup_table_array = slice_data(fuel_data, start_row_index, start_row_index + 20, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array_sliced = slice_data(fuel_data, start_row_index, start_row_index, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array = match_lookup_array_sliced[0]
        main_vlookup_col_index = MATCH(round(data['MR'] / 0.25, 0) * 0.25, match_lookup_array, 0)
        data["c*"] = (float(VLOOKUP(outer_vlookup_lookup_value, main_vlookup_table_array, main_vlookup_col_index, True)) + data['c*']) / 2
      if i == 1:
        data['╢D1'] = 0.75
      elif i >= 2:
        if data['N2O Status'] == "Gas" and calculations[i-2]['N2O Status'] == "Liquid":
          data['╢D1'] = 1
        else:
          if data['Pc'] < calculations[i-2]['Pc']:
            data['╢D1'] = data['Pc'] / calculations[i-2]['Pc']
          else:
            data['╢D1'] = 1
      if i > 0:
        data['Pc_raw'] = c_star_efficiency * data['c*'] * data['m_dot'] / throat_area / 1000
        data['dP/dt'] = (data['Pc_raw'] - data['Pc']) / data['Step']
      if i < 10 and i > 0:
        if data['m_ox'] > 0 and data['m_f'] > 0:
          if data['N2O Status'] == "Gas" and calculations[i-2]['N2O Status'] == "Liquid":
            data['Pc'] = (data['Pc'] / 2) * data['╢D1'] * (0.1 * i)
          else:
            data['Pc'] = data['Pc_raw'] * data['╢D1'] * (0.1 * i)
        else:
          data['Pc'] = None
          break
      elif i >= 10 and i < 1 and i < 15: # This condition 'i < 1' makes this block effectively unreachable for i >= 10.
                                          # Assuming it meant 'i < max_sim_time' or similar logical block for this range.
                                          # Replicated as-is from original code.
        sum = 0
        if data['m_ox'] > 0 and data['m_f'] > 0:
          if data['N2O Status'] == "Gas" and calculations[i-2]['N2O Status'] == "Liquid":
            sum = (data['Pc_raw'] + calculations [i-2]['Pc']) / 2 * data['╢D1']
          else:
            sum = data['Pc_raw'] * data['╢D1']
          data['Pc'] = (sum + data['Pc']) / 2
        else:
          data['Pc'] = None
          break
      elif i >= 15:
        sum = 0
        if data['m_ox'] > 0 and data['m_f'] > 0:
          if data['N2O Status'] == "Gas" and calculations[i-2]['N2O Status'] == "Liquid":
            sum = (data['Pc_raw'] + data['Pc']) / 2
          else:
            if SIGN(data['dP/dt']) != SIGN(calculations[i-1]['dP/dt']) and SIGN(data['dP/dt']) == SIGN(calculations[i-2]['dP/dt']) and SIGN(data['dP/dt']) != SIGN(calculations[i-3]['dP/dt']):
              sum = (data['Pc_raw'] + data['Pc']) / 2
            else:
              sum = data['Pc_raw']
        else:
          data['Pc'] = None
          break
        if data['Pc'] is not None:
          data['Pc'] = ((sum * data['╢D1']) + data['Pc_raw']) / 2
      if i == 0:
        data['Pc_psi'] = data['Pc'] / 6.89475729
      else:
        if data['m_ox'] > 0 and data['m_f'] > 0:
          data['Pc_psi'] = CONVERT(data['Pc'] * 1000, "Pa", "psi")
        else:
          data['Pc_psi'] = None
          break
      if i == 0:
        data['ΔP_ox'] = data['P_N2O_inj'] - data['Pc']
      elif i < 10 or i >= 15:
        data['ΔP_ox'] = data['P_N2O_inj'] - data['Pc'] if data['m_ox'] > 0 and data['m_f'] > 0 else None
      elif i < 15:
        data['ΔP_ox'] = ((data['P_N2O_inj'] - data['Pc']) + data['ΔP_ox']) / 2  if data['m_ox'] > 0 and data['m_f'] > 0 else None
      data['ΔP_ox (psi)'] = CONVERT(data['ΔP_ox'] * 1000, "Pa", "psi")
      if i == 0:
        data['ΔP_fu'] = data['P_fu_inj'] - data['Pc']
      elif i < 10 or i >= 15:
        data['ΔP_fu'] = data['P_fu_inj'] - data['Pc'] if data['m_ox'] > 0 and data['m_f'] > 0 else None
      elif i < 15:
        data['ΔP_fu'] = ((data['P_fu_inj'] - data['Pc']) + data['ΔP_fu']) / 2  if data['m_ox'] > 0 and data['m_f'] > 0 else None
      data['ΔP_fu (psi)'] = CONVERT(data['ΔP_fu'] * 1000, "Pa", "psi")
      if i == 1:
        data['Me'] = 1
      elif i > 1:
        slice_start_col = min(MATCH(data['gam0'], isentropic_flow_data[0], 1), MATCH(data['gam0'], isentropic_flow_data[0], 1) - 5)
        slice_end_col = max(MATCH(data['gam0'], isentropic_flow_data[0], 1), MATCH(data['gam0'], isentropic_flow_data[0], 1) - 5)
        main_vlookup_table_array = slice_data(isentropic_flow_data, 4, 503, slice_start_col, slice_end_col)
        data['Me'] = float(VLOOKUP(expansion_ratio, main_vlookup_table_array, 6, True))
      if i == 0:
        data['Pe'] = data['Pc'] / 10
      elif i == 1:
        data['Pe'] = data['Pc'] / 100
      else:
        data['Pe'] = data['Pc'] /(1 + (data['gam0'] - 1) / 2 * data['Me']**2)**(data['gam0'] / (data['gam0'] - 1))
      data['Pe_psi'] = CONVERT(data['Pe'] * 1000, "Pa", "psi")
      if i == 0:
        data['rho_o'] = float(VLOOKUP(data['P_N2O_tank'], slice_data(n2o_data, 3, 231, "B", "N"), 2, True))
      else:
        data['rho_o'] = float(VLOOKUP(data['P_N2O_tank'], slice_data(n2o_data, 3, 231, "B", "N"), 2, True)) if data['N2O Status'] == "Liquid" else float(VLOOKUP(data['P_N2O_tank'], slice_data(n2o_data, 3, 231, "R", "AD"), 2, True))
      if i == 0:
        data['m_dot_o'] = oxidizer_CdA * math.sqrt(2 * data['rho_o'] * (data['ΔP_ox']) * 1000)
      else:
        data['m_dot_o'] = oxidizer_CdA * math.sqrt(2 * data['rho_o'] * (data['ΔP_ox']) * 1000) if (oxidizer_CdA * math.sqrt(2 * data['rho_o'] * (data['ΔP_ox']) * 1000)) < data['m_dot_o'] and data['N2O Status'] == "Gas" else (oxidizer_CdA * math.sqrt(2 * data['rho_o'] * (data['ΔP_ox']) * 1000) + data['m_dot_o']) / 2
      data['Q_o'] = data['m_dot_o'] / data['rho_o']
      if i == 0:
        data['µ_o'] = float(VLOOKUP(data['P_N2O_tank'], slice_data(n2o_data, 3, 231, "B", "N"), 11, True)) / 1000
      else:
        data['µ_o'] = float(VLOOKUP(data['P_N2O_tank'], slice_data(n2o_data, 3, 231, "B", "N"), 11, True)) / 1000 if data['N2O Status'] == "Liquid" else float(VLOOKUP(data['P_N2O_tank'], slice_data(n2o_data, 3, 231, "R", "AD"), 12, True)) / 1000
      data['v_o_line'] = data['m_dot_o'] / (data['rho_o'] * ox_feed_line_area)
      data['Re_o'] = data['rho_o'] * data['v_o_line'] * CONVERT(ox_feed_line_diameter, ox_feed_line_diameter_units, "m") / data['µ_o']
      if i == 0:
        data['cgd_o'] = -2 * math.log10(2.51 / (data['Re_o'] * 0.01**0.5) + (CONVERT(ox_line_roughness, ox_line_roughness_units, "m") / CONVERT(ox_feed_line_diameter, ox_feed_line_diameter_units, "m") / 3.72))
      else:
        data['cgd_o'] = -2 * math.log10(2.51 / (data['Re_o'] * data['dwc_o']**0.5) + (CONVERT(ox_line_roughness, ox_line_roughness_units, "m") / CONVERT(ox_feed_line_diameter, ox_feed_line_diameter_units, "m") / 3.72))
      data['dwc_o'] = 1 / data['cgd_o']**2
      data['Q_o_gpm'] = data['Q_o'] * 15850.3
      data['dP_o_Cv'] = 0 if ox_component_Cv == 0 else CONVERT((data['Q_o_gpm'] / ox_component_Cv)**2 * data['rho_o'] / 1000, "psi", "pa")
      data['dP_o_line'] = data['dwc_o'] * (CONVERT(ox_line_length, ox_line_length_units, "m") / CONVERT(ox_feed_line_diameter, ox_feed_line_diameter_units, "m")) * (data['rho_o'] * data['v_o_line']**2 / 2)
      if i == 0 or i >= 10:
        data['dP_o_total'] = data['dP_o_Cv'] + data['dP_o_line'] + CONVERT(ox_additional_loss, "psi", "pa")
      else:
        data['dP_o_total'] = (data['dP_o_Cv'] + data['dP_o_line'] + CONVERT(ox_additional_loss, "psi", "pa")) * (0.1 * i)
      data['dP_o_total_psi'] = CONVERT(data['dP_o_total'], "Pa", "psi")
      data['rho_f'] = fuel_density
      if i == 0:
        data['m_dot_f'] = fuel_CdA * math.sqrt(2 * data['rho_f'] * (data['P_fu_inj'] - data['Pc']) * 1000)
      else:
        data['m_dot_f'] = fuel_CdA * math.sqrt(2 * data['rho_f'] * data['ΔP_fu'] * 1000) if (fuel_CdA * math.sqrt(2 * data['rho_f'] * data['ΔP_fu'] * 1000)) < data['m_dot_f'] and data['N2O Status'] == "Gas" else (fuel_CdA * math.sqrt(2 * data['rho_f'] * data['ΔP_fu'] * 1000) + data['m_dot_f']) / 2
      data['Q_f'] = data['m_dot_f'] / float(VLOOKUP(fuel, slice_data(fuel_data,4,14,"E","H"), 3, False))
      data['v_f_line'] = data['m_dot_f'] / (float(VLOOKUP(fuel, slice_data(fuel_data,4,14,"E","H"), 3, False)) * f_feed_line_area)
      data['Re_f'] = float(VLOOKUP(fuel, slice_data(fuel_data,4,14,"E","H"), 3, False)) * data['v_f_line'] * CONVERT(f_feed_line_diameter, f_feed_line_diameter_units, "m") / float(VLOOKUP(fuel, slice_data(fuel_data,4,14,"E","H"), 4, False))
      if i == 0:
        data['cgd_f'] = -2 * math.log10(2.51 / (data['Re_f'] * 0.025**0.5) + (CONVERT(f_line_roughness, f_line_roughness_units, "m") / CONVERT(f_feed_line_diameter, f_feed_line_diameter_units, "m") / 3.72))
      else:
        data['cgd_f'] = -2 * math.log10(2.51 / (data['Re_f'] * data['dwc_f']**0.5) + (CONVERT(f_line_roughness, f_line_roughness_units, "m") / CONVERT(f_feed_line_diameter, f_feed_line_diameter_units, "m") / 3.72))
      data['dwc_f'] = 1 / data['cgd_f']**2
      data['Q_f_gpm'] = data['Q_f'] * 15850.3
      data['dP_f_Cv'] = 0 if f_component_Cv == 0 else CONVERT((data['Q_f_gpm'] / f_component_Cv)**2 * data['rho_f'] / 1000, "psi", "pa")
      data['dP_f_line'] = data['dwc_f'] * (CONVERT(f_line_length, f_line_length_units, "m") / CONVERT(f_feed_line_diameter, f_feed_line_diameter_units, "m")) * (data['rho_f'] * data['v_f_line']**2 / 2)
      if i == 0 or i >= 10:
        data['dP_f_total'] = data['dP_f_Cv'] + data['dP_f_line'] + CONVERT(f_additional_loss, "psi", "pa")
      elif i < 10:
        data['dP_f_total'] = (data['dP_f_Cv'] + data['dP_f_line'] + CONVERT(f_additional_loss, "psi", "pa")) * (i * 0.1)
      data['dP_f_total_psi'] = CONVERT(data['dP_f_total'], "Pa", "psi")
      data['m_dot'] = data['m_dot_o'] + data['m_dot_f']
      if i == 0:
        data['m_ox'] = oxidizer_volume * n2o_density / 61024
      else:
        data['m_ox'] = 0 if (data['m_ox'] - data['m_oxgen']) < 0 else data['m_ox'] - data['m_oxgen']
      data['m_oxgen'] = 0 if data['m_ox'] == 0 or data['m_oxgen'] == 0 or data['Status'] == "OD" else data['m_dot_o'] * data['Step']
      if i == 0:
        data['m_fgen'] = data['m_dot_f'] * data['Step']
      else:
        data['m_fgen'] = 0 if data['m_f'] == 0 else data['m_dot_f'] * data['Step']
      if i == 0:
        data['m_f'] = fuel_volume * fuel_density / 61024
      else:
         data['m_f'] = 0 if (data['m_f'] - data['m_fgen']) < 0 else data['m_f'] - data['m_fgen']
      if data['m_dot_o'] == 0 or data['m_dot_f'] == 0:
        if data['MR'] != 0:
          data['MR'] = 0
        else:
          data['MR'] = None
          break
      else:
        data['MR'] = data['m_dot_o'] / data['m_dot_f']
      if i == 1:
        data['T0'] = 1000
      elif i > 1:
        vlookup_lookup_value = 50 if (round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50) == 0 else round(data['Pc_psi'] / c_star_efficiency / 50, 0) * 50
        start_row_index = int(VLOOKUP("T0", slice_data(fuel_data, 4, 7, 'B', 'C'), 2, False))
        start_col_excel_num = int(VLOOKUP(fuel, slice_data(fuel_data, 4, 38, 'E', 'F'), 2, False))
        main_vlookup_table_array = slice_data(fuel_data, start_row_index, start_row_index + 20, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array_sliced = slice_data(fuel_data, start_row_index, start_row_index, start_col_excel_num, start_col_excel_num + 40)
        match_lookup_array = match_lookup_array_sliced[0]
        vlookup_col_index = MATCH(round(data['MR'] / 0.25, 0) * 0.25, match_lookup_array, 0)
        inner_vlookup_result = VLOOKUP(vlookup_lookup_value, main_vlookup_table_array, vlookup_col_index, True)
        data['T0'] = (c_star_efficiency**2) * float(inner_vlookup_result)
      data['Te'] = data['T0'] / (1 + (data['gam0'] - 1) / 2 * data['Me']**2)
      if i  < 2:
        data['ve'] = data['Me'] * math.sqrt(data['gam0'] * data['R'] * data['Te'])
      else:
        data['ve'] = nozzle_efficiency * data['Me'] * math.sqrt(data['gam0'] * data['R'] * data['Te'])
      if i == 0:
        data['╢D1_1'] = 1
      elif i == 1:
        data['╢D1_1'] = 0.75
      else:
        data['╢D1_1'] = data['F'] / calculations[i-2]['F'] if data['F'] < calculations[i-2]['F'] else 1
      if i == 0:
        data['╢D2'] = 1
      elif i < 15:
        data['╢D2'] = 1500000 * data['Step']
      else:
        data['╢D2'] = 1
      if i < 2:
        data['F_raw'] = data['m_dot'] * data['ve']
      else:
        if data['F'] < 1 or data['Status'] == "FD" or data['Status'] == "OD":
          data['F_raw'] = 0
        else:
          common_calculation = data['m_dot'] * data['ve'] * (1 + math.cos(math.radians(exit_half_angle))) / 2 + (data['Pe'] * 1000 - CONVERT(ambient_pressure, "psi", "Pa")) * exit_area
          if common_calculation < 0.01:
            data['F_raw'] = 0.01
          else:
            data['F_raw'] = common_calculation
      if i > 0:
         data['dF/dt'] = (data['F_raw'] - data['F']) / data['Step']
      if i == 0:
        data['F'] = data['F_raw']
      elif i < 10:
        data['F'] = (data['╢D2'] * data['Step'] + data['F'] if ((data['F_raw'] - data['F']) / data['Step']) > data['╢D2'] else data['F_raw']) * data['╢D1_1']
      else:
        sum = 0
        if data['m_ox'] > 0 and data['m_f'] > 0:
          if data['N2O Status'] == "Gas" and calculations[i-2]['N2O Status'] == "Liquid":
            sum = (data['F_raw'] + data['F']) / 2
          else:
            if SIGN(data['dF/dt']) != SIGN(calculations[i-1]['dF/dt']) and SIGN(data['dF/dt']) == SIGN(calculations[i-2]['dF/dt']) and SIGN(data['dF/dt']) != SIGN(calculations[i-3]['dF/dt']):
              sum = (data['F_raw'] + data['F']) / 2
            else:
              sum = data['F_raw']
        else:
          data['F'] = None
          break
        if data['F'] is not None:
          data['F'] = ((sum * data['╢D1_1']) + data['F']) / 2
      data['F_lbf'] = CONVERT(data['F'], "N", "lbf")
      data['J'] = data['F'] * data['Step']
      if i == 0:
        data['J_deliv'] = 0 if data['Time'] == 0 else data['J']
      else:
        data['J_deliv'] = 0 if data['Time'] == 0 else data['J'] + data['J_deliv']
      if data['Status'] == "FD" or data['Status'] == "OD":
        data['Status'] = data['Status']
      else:
        if data['m_f'] == 0:
          data['Status'] = "FD"
        else:
          if data['m_ox'] == 0:
            data['Status'] = "OD"
          else:
            data['Status'] = "Burning"
      if i < 2:
        data['N2O Status'] = "Liquid"
      else:
        data['N2O Status'] = "Liquid" if data['m_ox'] / init_m_ox > 0.1 else "Gas"
      if i == 0:
        init_P_N2O_tank = data['P_N2O_tank']
        init_m_ox = data['m_ox']
      calculations.append(data.copy())
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) # Try a large number like 1000, 2000, or more if needed
    pd.set_option('display.colheader_justify', 'left') # Optional: helps align headers

    df_calculations = pd.DataFrame(calculations)

    # results
    exit_pressure = calculate_result(df_calculations, "Pe_psi", 7, 14)
    flow_condition = "IDEALLY" if abs(ambient_pressure - exit_pressure) < (ambient_pressure * 0.025) else ("OVER" if exit_pressure < ambient_pressure else "UNDER")
    flow_separation = "NO" if df_calculations.loc[5, "Pe_psi"] > (ambient_pressure / 2) else ("POSSIBLE" if df_calculations.loc[5, "Pe_psi"] > (ambient_pressure * 0.4) else "PROBABLE")
    chamber_pressure = INTERCEPT(df_calculations, "Pc_psi", "Time", 28, 33)
    initial_thrust = INTERCEPT(df_calculations, "F_lbf", "Time", 28, 33)
    burn_time = df_calculations.iloc[-1,0]
    mixture_ratio = INTERCEPT(df_calculations, "MR", "Time", 28, 33)
    mass_flow_rate = INTERCEPT(df_calculations, "m_dot", "Time", 28, 33)
    overall_efficieny = nozzle_efficiency * c_star_efficiency * (1 + math.cos(math.radians(exit_half_angle))) / 2
    tank_ratio = CONVERT(oxidizer_mass, oxidizer_mass_units, "kg") / CONVERT(fuel_mass, fuel_mass_units, "kg")
    oxidizer_injection_delta = CONVERT(df_calculations[df_calculations['ΔP_ox'] > 0]['ΔP_ox'].min() * 1000, "Pa", "psi")
    oxidizer_stiffness = oxidizer_injection_delta / chamber_pressure
    fuel_injection_delta = CONVERT(df_calculations[df_calculations['ΔP_fu'] > 0]['ΔP_fu'].min() * 1000, "Pa", "psi")
    fuel_stiffness = fuel_injection_delta / chamber_pressure
    total_impulse = df_calculations['J'].sum()
    specific_impulse_S = CONVERT(initial_thrust, "lbf", "N") / ((mass_flow_rate) * 9.81)
    specific_impulse_C = total_impulse / ((CONVERT(fuel_mass, fuel_mass_units, "kg") + CONVERT(oxidizer_mass, oxidizer_mass_units, "kg")) * 9.81)

    return {
        "exit_pressure": exit_pressure,
        "flow_condition": flow_condition,
        "flow_separation": flow_separation,
        "chamber_pressure": chamber_pressure,
        "initial_thrust": initial_thrust,
        "burn_time": burn_time,
        "mixture_ratio": mixture_ratio,
        "mass_flow_rate": mass_flow_rate,
        "overall_efficiency": overall_efficieny,
        "tank_ratio": tank_ratio,
        "oxidizer_injection_delta": oxidizer_injection_delta,
        "oxidizer_stiffness": oxidizer_stiffness,
        "fuel_injection_delta": fuel_injection_delta,
        "fuel_stiffness": fuel_stiffness,
        "total_impulse": total_impulse,
        "specific_impulse_S": specific_impulse_S,
        "specific_impulse_C": specific_impulse_C
    }
    
# Define the paths to your CSV files (replace with actual paths if different)
N2O_PATH = "N2O.csv"
FUEL_PATH = "Fuel.csv"
FORMULAS_PATH = "formulas.csv"
CALCULATIONS_PATH = "Calculations.csv"
ISENTROPIC_FLOW_PATH = "Isentropic Flow.csv"

# Define ranges for the engine dimensions to optimize
# These ranges are illustrative. Adjust them based on your design constraints and desired exploration.
throat_diameter_values = [0.8, 1.0, 1.2] # inches
exit_diameter_values = [1.8, 2.0, 2.2] # inches (should generally be larger than throat_diameter)
chamber_diameter_values = [1.8, 2.0, 2.2] # inches (should generally be larger than throat_diameter)
chamber_length_values = [4.0, 5.0, 6.0] # inches
exit_half_angle_values = [10, 15, 20] # degrees

best_impulse = -1.0
best_parameters = {}
best_results = {}

print("Starting iterative design optimization for engine dimensions...\n")

total_iterations = (
    len(throat_diameter_values) *
    len(exit_diameter_values) *
    len(chamber_diameter_values) *
    len(chamber_length_values) *
    len(exit_half_angle_values)
)
current_iteration = 0

for td in throat_diameter_values:
    for ed in exit_diameter_values:
        for cd in chamber_diameter_values:
            for cl in chamber_length_values:
                for eha in exit_half_angle_values:
                    current_iteration += 1
                    print(f"Iteration {current_iteration}/{total_iterations}: "
                          f"TD={td:.2f} in, ED={ed:.2f} in, CD={cd:.2f} in, CL={cl:.2f} in, EHA={eha:.1f}°")

                    # Ensure exit_diameter is greater than throat_diameter for a valid nozzle
                    if ed <= td:
                        print("  -> Skipping: Exit diameter must be greater than throat diameter.")
                        continue
                    # Ensure chamber_diameter is greater than throat_diameter
                    if cd <= td:
                        print("  -> Skipping: Chamber diameter must be greater than throat diameter.")
                        continue

                    try:
                        # Run the simulation with the current set of parameters
                        current_results = run_simulation(
                            n2o_filepath=N2O_PATH,
                            fuel_filepath=FUEL_PATH,
                            formulas_filepath=FORMULAS_PATH,
                            calculations_filepath=CALCULATIONS_PATH,
                            isentropic_flow_filepath=ISENTROPIC_FLOW_PATH,
                            throat_diameter=td,
                            exit_diameter=ed,
                            chamber_diameter=cd,
                            chamber_length=cl,
                            exit_half_angle=eha
                            # All other parameters will use their default values from run_simulation
                        )

                        if current_results and current_results.get('total_impulse') is not None:
                            current_impulse = current_results['total_impulse']
                            print(f"  -> Total Impulse: {current_impulse:.2f} N")

                            # Check if this is the best thrust found so far
                            if current_impulse > best_impulse:
                                best_impulse = current_impulse
                                best_parameters = {
                                    "throat_diameter": td,
                                    "exit_diameter": ed,
                                    "chamber_diameter": cd,
                                    "chamber_length": cl,
                                    "exit_half_angle": eha,
                                    "total_impulse": current_impulse,
                                }
                                best_results = current_results
                                print(f"  *** NEW BEST FOUND: {best_impulse:.2f} lbf ***")
                        else:
                            print("  -> Simulation failed or returned no initial thrust for these parameters.")
                    except Exception as e:
                        print(f"  -> An error occurred during simulation: {e}")
                        print("  -> Skipping this parameter combination.")
                    print("-" * 50) # Separator for readability

print("\n--- Optimization Complete ---")
if best_parameters:
    print(f"Optimal Engine Dimensions Found:")
    print(f"  Throat Diameter: {best_parameters['throat_diameter']:.3f} in")
    print(f"  Exit Diameter: {best_parameters['exit_diameter']:.3f} in")
    print(f"  Chamber Diameter: {best_parameters['chamber_diameter']:.3f} in")
    print(f"  Chamber Length: {best_parameters['chamber_length']:.2f} in")
    print(f"  Exit Half Angle: {best_parameters['exit_half_angle']:.1f}°")
    print(f"  Maximum Initial Thrust: {best_parameters['total_impulse']:.2f} lbf")
    for key, value in best_results:   
        print(f"{key}: {value}")
else:
    print("No optimal design found. This could be due to: \n"
          "1. Incorrect CSV file paths.\n"
          "2. All simulations failing (check error messages).\n"
          "3. The defined parameter ranges not yielding any valid thrust values.")
