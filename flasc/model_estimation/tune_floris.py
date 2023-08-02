# Copyright 2023 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import os
import copy
import yaml

import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from flasc.model_estimation import floris_params_model
from flasc import floris_tools as ftools
from flasc.energy_ratio import energy_ratio_suite
from flasc.dataframe_operations import dataframe_manipulations as dfm

from floris.tools import FlorisInterface
from floris.tools import ParallelComputingInterface


class FlorisTuner():
    """
    Given a FLORIS model, FlorisTuner provides a suite of methods for tuning said model,
    extracting the updated parameters and writing said parameters to a YAML file. 

    Args:
        df_scada (:py:obj:`pd.DataFrame`): SCADA data.

        num_turbines (:py:obj:`int`): Number of turbines in the wind farm.

        test_turbines (:py:obj:`int`): Index of test turbines.

        steered_turbine (:py:obj:`int`): Index of wake steered turbine. # TODO: Check if there should be an option for steering multiple turbines

        yaw (:py:obj:`float`): Yaw angle(s) for wake steering.

        breakpoints_D (:py:obj:`float`): Diameter constant. 

    """

    def __init__(self, 
                 fi: FlorisInterface,
                 df_scada: pd.DataFrame, 
                 num_turbines: int, 
                 test_turbines: int, 
                 steered_turbine: int = None, 
                 yaw: list[float] = None, 
                 breakpoints_D: float = None):

        self.fi_untuned = fi

        self.fi_tuned = None # initialize for later tuning
        
        self.df_scada = df_scada

        self.num_turbines = num_turbines
        
        self.test_turbines = test_turbines

        self.steered_turbine = steered_turbine
  
        self.yaw = yaw
        
        self.breakpoints_D = breakpoints_D
  
        self.yaml = '' # initialize for later yaml file writing

    def get_floris_df(self, 
                      fi: FlorisInterface, 
                      pow_ref_columns: list[int], 
                      time_series: bool = True) -> pd.DataFrame:
        """
        Generate dataframe for FLORIS predictions for comparison with SCADA.

        Args:
            fi (:py:obj:`FlorisInterface`): FLORIS model.

            pow_ref_columns (:py:obj:`list[int]`): Reference power columns that should be used for generating FLORIS predictions.

            time_series (:py:obj:`bool`): Specify if time series should be used for generating FLORIS predictions. Defaults to True.

        Returns:
            df_floris (:py:obj:`pd.DataFrame`): FLORIS predictions to be compared with SCADA.

        """
    
        # Copy FLORIS model
        fi_ = copy.deepcopy(fi)

        # Get wind speeds and directions from SCADA
        wind_speeds = self.df_scada['ws']
        wind_directions = self.df_scada['wd']

        # If wake steering case, apply yaw angles to the steered turbine and factor that into the wake calculation
        if self.yaw is not None:
            # If time series is set to True, set the wind speed dimension to 1
            if time_series:
                yaw_angles = np.zeros([len(wind_directions),1,self.num_turbines])
                yaw_angles[:,0, self.steered_turbine] = self.yaw

                fi_.reinitialize(wind_speeds=wind_speeds, 
                                wind_directions=wind_directions,
                                time_series=time_series)

                fi_.calculate_wake(yaw_angles=yaw_angles)
            # If time series is set to False, set the wind speed dimension to the number of wind speeds
            else:
                yaw_angles = np.zeros([len(wind_directions),len(wind_speeds),self.num_turbines])
                yaw_angles[:,0, self.steered_turbine] = self.yaw
            
                fi_.reinitialize(wind_speeds=wind_speeds, 
                                wind_directions=wind_directions,
                                time_series=time_series)

                fi_.calculate_wake(yaw_angles=yaw_angles)

        # If baseline (non-wake steering) case, skip yaw angles in the wake calculation
        else:
            fi_.reinitialize(wind_speeds=wind_speeds, 
                             wind_directions=wind_directions,
                             time_series=time_series)
            
            fi_.calculate_wake()

        # Normalize turbine powers
        turbine_powers = fi_.get_turbine_powers().squeeze()/1000.

        # Generate FLORIS DataFrame
        df_floris = pd.DataFrame(data=turbine_powers,
                                 columns=[f"pow_{str(i).zfill(3)}" for i in range(self.num_turbines)])
        
        df_floris = df_floris.assign(ws=wind_speeds,
                                     wd=wind_directions,
                                     pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

        return df_floris

    def evaluate_error(self, 
                       case: str,
                       df_floris: pd.DataFrame, 
                       wd_step: float = 3.0,
                       ws_step: float = 5.0,
                       wd_bin_width: float = None,
                       wd_bins: npt.NDArray[np.float64] = None,
                       ws_bins: npt.NDArray[np.float64] = None,
                       N: int = 1,
                       percentiles: list[float] = [5.0, 95.0],
                       balance_bins_between_dfs: bool = True,
                       return_detailed_output: bool = False,
                       num_blocks: int = -1,
                       verbose: bool = True) -> float:
        """
        Compares SCADA and FLORIS energy ratios and evaluates their error (mean squared error).

        Args:
            df_floris (:py:obj:`pd.DataFrame`): FLORIS predictions to be compared with SCADA.

            wd_step (:py:obj:`float`): Wind direction discretization step size. 
                    Specifies which wind directions the energy ratio will be calculated for. 
                    Note that that not each bin has a width of this value. 
                    This variable is ignored if 'wd_bins' is specified. 
                    Defaults to 3.0.

            ws_step (:py:obj:`float`): Wind speed discretization step size.
                    Specifies the resolution and width of the wind speed bins. 
                    This variable is ignored if 'ws_bins' is specified. 
                    Defaults to 5.0.

            wd_bin_width (:py:obj:`float`): Wind direction bin width.
                    Should be equal or larger than 'wd_step'. 
                    Note that in the literature, it is not uncommon to 
                    specify a bin width that is larger than the step size 
                    to cover for variability in the wind direction measurements. 
                    By setting a large value for 'wd_bin_width', this provides a 
                    perspective of the larger-scale wake losses in the wind farm. 
                    When no value is provided, 'wd_bin_width' = 'wd_step'.
                    Defaults to None.

            wd_bins (:py:obj:`np.array[float]` arr): Wind direction bins over which the energy 
                    ratio will be calculated. Each entry of this array must contain exactly two 
                    float values, the lower and upper bound for that wind direction bin. 
                    Overlap between bins is supported.
                    This variable overwrites the settings for 'wd_step' and 'wd_bin_width', and 
                    instead allows the user to directly specify the binning properties, rather than 
                    deriving them from the data and an assigned step size and bin width.
                    Defaults to None.

            ws_bins (:py:obj:`np.array[float]` arr): Wind speed bins over which the energy 
                    ratio will be calculated. Each entry of this array must contain exactly 
                    two float values, the lower and upper bound for that wind speed bin.
                    Overlap between bins is not currently supported.
                    This variable overwrites the settings for 'ws_step', and instead allows 
                    the user to directly specify the binning properties, rather than deriving 
                    them from the data and an assigned step size. 
                    Defaults to None.
            
            N (:py:obj:`int`): Number of bootstrap evaluations for uncertainty quantification (UQ). 
                    If N = 1, no UQ will be performed. 
                    Defaults to 1.

            percentiles (:py:obj:`list[float]`): Confidence bounds for uncertainty quantification (UQ) in percents. 
                    This value is only relevant if N > 1 is specified. 
                    Defaults to [5.0, 95.0].

            balance_bins_between_dfs (:py:obj:`bool`): Balances the bins by the frequency of occurrence for each 
                    wind direction and wind speed bin in the collective of dataframes. The frequency of a
                    certain bin is equal to the minimum number of occurrences among all the dataframes. 
                    This ensures an "apples to apples" comparison. Recommended to set to 'True'. 
                    Will avoid bin rebalancing if the underlying 'wd' and 'ws' occurrences are identical 
                    between all dataframes (i.e. comparing SCADA data to FLORIS predictions of the same data).
                    Defaults to True.

            return_detailed_output (:py:obj:`bool`): Calculates and returns detailed energy ratio information useful for debugging
                    and evaluating flaws in the data. This can impact the speed of calculations but can be very useful. 
                    This information is written to 'self.df_lists[i]['er_results_info_dict']'. The dictionary contains 
                    two fields, 'df_per_wd_bin' and 'df_per_ws_bin'. 'df_per_wd_bin' provides an overview of the energy ratio for 
                    every wind direction bin, covering the collective effect of all wind speeds in the data. 'df_per_ws_bin' provides 
                    more information and displays the energy ratio for every wind direction and wind speed bin, among others. 
                    This is particularly helpful in determining if the bins are well balanced. 
                    Defaults to False.

            num_blocks (:py:obj:`int`): Number of blocks to use in block boostrapping. 
                    If num_blocks = -1, then do not use block bootstrapping and follow the 
                    normal approach of randomly sampling 'num_samples' with replacement.  
                    Defaults to -1.

            verbose (:py:obj:`bool`): Specify printing to console. 
                    Defaults to True.


        Returns:
            err (:py:obj:`float`): Flag indicating if the SCADA and FLORIS energy ratios are equal within the given tolerance. 
                    Used to determine if further tuning of FLORIS is needed.

        """
        # Validate 'case'
        if case != 'baseline' and case != 'controlled':
                raise ValueError("Can only evaluate the 'baseline' or 'controlled' case.")

        # Generate energy ratios for SCADA and FLORIS
        s = energy_ratio_suite.energy_ratio_suite()
        s.add_df(self.df_scada, name=f"SCADA ({case})", color='b')
        s.add_df(df_floris, name=f"FLORIS ({case})", color='r')

        energy_ratios = s.get_energy_ratios(test_turbines=self.test_turbines,
                                            wd_step=wd_step,
                                            ws_step=ws_step,
                                            wd_bin_width=wd_bin_width,
                                            wd_bins=wd_bins,
                                            ws_bins=ws_bins,
                                            N=N,
                                            percentiles=percentiles,
                                            balance_bins_between_dfs=balance_bins_between_dfs,
                                            return_detailed_output=return_detailed_output,
                                            num_blocks=num_blocks,
                                            verbose=verbose)

        # Take mean squared error (mse) of the energy ratios
        # TODO: The 'baseline' key for energy ratios is confusing for the 'controlled' case and needs to be updated
        scada_energy_ratios = energy_ratios[0]['er_results']['baseline']   
        floris_energy_ratios = energy_ratios[1]['er_results']['baseline']

        datapoints_per_wd_bin = energy_ratios[0]['er_results']['bin_count']

        err = mean_squared_error(y_true=scada_energy_ratios, 
                                 y_pred=floris_energy_ratios, 
                                 sample_weight=datapoints_per_wd_bin)

        return err

    def visualize_scada_comparison(self,
                                   case: str,
                                   title: str,
                                   pow_ref_columns: list[int],
                                   time_series: bool = True,
                                   wd_step: float = 3.0,
                                   ws_step: float = 5.0,
                                   wd_bin_width: float = None,
                                   wd_bins: npt.NDArray[np.float64] = None,
                                   ws_bins: npt.NDArray[np.float64] = None,
                                   N: int = 1,
                                   percentiles: list[float] = [5.0, 95.0],
                                   balance_bins_between_dfs: bool = True,
                                   return_detailed_output: bool = False,
                                   num_blocks: int = -1,
                                   verbose: bool = True):
        """
        Generate a visualization comparing SCADA, untuned and tuned FLORIS energy ratios.

        Args:
            case (:py:obj:`str`): Case being analyzed. Either "baseline" or "controlled".

            title (:py:obj:`str`): Title of visualization. 

            pow_ref_columns (:py:obj:`list[int]`): Reference power columns that should be used for generating FLORIS predictions.            
            
            time_series (:py:obj:`bool`): Specify if time series should be used for generating FLORIS predictions. Defaults to True.

            wd_step (:py:obj:`float`): Wind direction discretization step size. 
                    Specifies which wind directions the energy ratio will be calculated for. 
                    Note that that not each bin has a width of this value. 
                    This variable is ignored if 'wd_bins' is specified. 
                    Defaults to 3.0.

            ws_step (:py:obj:`float`): Wind speed discretization step size.
                    Specifies the resolution and width of the wind speed bins. 
                    This variable is ignored if 'ws_bins' is specified. 
                    Defaults to 5.0.

            wd_bin_width (:py:obj:`float`): Wind direction bin width.
                    Should be equal or larger than 'wd_step'. 
                    Note that in the literature, it is not uncommon to 
                    specify a bin width that is larger than the step size 
                    to cover for variability in the wind direction measurements. 
                    By setting a large value for 'wd_bin_width', this provides a 
                    perspective of the larger-scale wake losses in the wind farm. 
                    When no value is provided, 'wd_bin_width' = 'wd_step'.
                    Defaults to None.

            wd_bins (:py:obj:`np.array[float]` arr): Wind direction bins over which the energy 
                    ratio will be calculated. Each entry of this array must contain exactly two 
                    float values, the lower and upper bound for that wind direction bin. 
                    Overlap between bins is supported.
                    This variable overwrites the settings for 'wd_step' and 'wd_bin_width', and 
                    instead allows the user to directly specify the binning properties, rather than 
                    deriving them from the data and an assigned step size and bin width.
                    Defaults to None.

            ws_bins (:py:obj:`np.array[float]` arr): Wind speed bins over which the energy 
                    ratio will be calculated. Each entry of this array must contain exactly 
                    two float values, the lower and upper bound for that wind speed bin.
                    Overlap between bins is not currently supported.
                    This variable overwrites the settings for 'ws_step', and instead allows 
                    the user to directly specify the binning properties, rather than deriving 
                    them from the data and an assigned step size. 
                    Defaults to None.
            
            N (:py:obj:`int`): Number of bootstrap evaluations for uncertainty quantification (UQ). 
                    If N = 1, no UQ will be performed. 
                    Defaults to 1.

            percentiles (:py:obj:`list[float]`): Confidence bounds for uncertainty quantification (UQ) in percents. 
                    This value is only relevant if N > 1 is specified. 
                    Defaults to [5.0, 95.0].

            balance_bins_between_dfs (:py:obj:`bool`): Balances the bins by the frequency of occurrence for each 
                    wind direction and wind speed bin in the collective of dataframes. The frequency of a
                    certain bin is equal to the minimum number of occurrences among all the dataframes. 
                    This ensures an "apples to apples" comparison. Recommended to set to 'True'. 
                    Will avoid bin rebalancing if the underlying 'wd' and 'ws' occurrences are identical 
                    between all dataframes (i.e. comparing SCADA data to FLORIS predictions of the same data).
                    Defaults to True.

            return_detailed_output (:py:obj:`bool`): Calculates and returns detailed energy ratio information useful for debugging
                    and evaluating flaws in the data. This can impact the speed of calculations but can be very useful. 
                    This information is written to 'self.df_lists[i]['er_results_info_dict']'. The dictionary contains 
                    two fields, 'df_per_wd_bin' and 'df_per_ws_bin'. 'df_per_wd_bin' provides an overview of the energy ratio for 
                    every wind direction bin, covering the collective effect of all wind speeds in the data. 'df_per_ws_bin' provides 
                    more information and displays the energy ratio for every wind direction and wind speed bin, among others. 
                    This is particularly helpful in determining if the bins are well balanced. 
                    Defaults to False.

            num_blocks (:py:obj:`int`): Number of blocks to use in block boostrapping. 
                    If num_blocks = -1, then do not use block bootstrapping and follow the 
                    normal approach of randomly sampling 'num_samples' with replacement.  
                    Defaults to -1.

            verbose (:py:obj:`bool`): Specify printing to console. 
                    Defaults to True.

        """

        # Generate dataframes for both tuned and untuned FLORIS
        df_floris_untuned = self.get_floris_df(fi=self.fi_untuned,
                                               pow_ref_columns=pow_ref_columns,
                                               time_series=time_series)
        
        df_floris_tuned = self.get_floris_df(fi=self.fi_tuned,
                                             pow_ref_columns=pow_ref_columns,
                                             time_series=time_series)

        # Generate energy ratios for SCADA and FLORIS
        s = energy_ratio_suite.energy_ratio_suite()
        s.add_df(self.df_scada, name=f"SCADA ({case})", color='b')
        s.add_df(df_floris_untuned, name=f"Untuned FLORIS ({case})", color='r')
        s.add_df(df_floris_tuned, name=f"Tuned FLORIS ({case})", color='g')

        s.get_energy_ratios(test_turbines=self.test_turbines,
                                          wd_step=wd_step,
                                          ws_step=ws_step,
                                          wd_bin_width=wd_bin_width,
                                          wd_bins=wd_bins,
                                          ws_bins=ws_bins,
                                          N=N,
                                          percentiles=percentiles,
                                          balance_bins_between_dfs=balance_bins_between_dfs,
                                          return_detailed_output=return_detailed_output,
                                          num_blocks=num_blocks,
                                          verbose=verbose)
        
        # Plot energy ratios
        ax = s.plot_energy_ratios(show_barplot_legend=False)
        ax[0].set_title(title)

    def tune_floris(self, 
                    case: str,
                    tolerance: float, 
                    pow_ref_columns: list[int],
                    params: npt.NDArray[np.float64],
                    time_series: bool = True,
                    wd_step: float = 3.0,
                    ws_step: float = 5.0,
                    wd_bin_width: float = None,
                    wd_bins: npt.NDArray[np.float64] = None,
                    ws_bins: npt.NDArray[np.float64] = None,
                    N: int = 1,
                    percentiles: list[float] = [5.0, 95.0],
                    balance_bins_between_dfs: bool = True,
                    return_detailed_output: bool = False,
                    num_blocks: int = -1,
                    verbose: bool = True
                    ):
        """
        Tune FLORIS model. 

        Args:
            case (:py:obj:`str`): Case being analyzed. Either "baseline" or "controlled".
            
            tolerance (:py:obj:`float`): Tolerance for error evaluation.

            pow_ref_columns (:py:obj:`list[int]`): Reference power columns that should be used for generating FLORIS predictions.            
            
            time_series (:py:obj:`bool`): Specify if time series should be used for generating FLORIS predictions. Defaults to True.

            wd_step (:py:obj:`float`): Wind direction discretization step size. 
                    Specifies which wind directions the energy ratio will be calculated for. 
                    Note that that not each bin has a width of this value. 
                    This variable is ignored if 'wd_bins' is specified. 
                    Defaults to 3.0.

            ws_step (:py:obj:`float`): Wind speed discretization step size.
                    Specifies the resolution and width of the wind speed bins. 
                    This variable is ignored if 'ws_bins' is specified. 
                    Defaults to 5.0.

            wd_bin_width (:py:obj:`float`): Wind direction bin width.
                    Should be equal or larger than 'wd_step'. 
                    Note that in the literature, it is not uncommon to 
                    specify a bin width that is larger than the step size 
                    to cover for variability in the wind direction measurements. 
                    By setting a large value for 'wd_bin_width', this provides a 
                    perspective of the larger-scale wake losses in the wind farm. 
                    When no value is provided, 'wd_bin_width' = 'wd_step'.
                    Defaults to None.

            wd_bins (:py:obj:`np.array[float]` arr): Wind direction bins over which the energy 
                    ratio will be calculated. Each entry of this array must contain exactly two 
                    float values, the lower and upper bound for that wind direction bin. 
                    Overlap between bins is supported.
                    This variable overwrites the settings for 'wd_step' and 'wd_bin_width', and 
                    instead allows the user to directly specify the binning properties, rather than 
                    deriving them from the data and an assigned step size and bin width.
                    Defaults to None.

            ws_bins (:py:obj:`np.array[float]` arr): Wind speed bins over which the energy 
                    ratio will be calculated. Each entry of this array must contain exactly 
                    two float values, the lower and upper bound for that wind speed bin.
                    Overlap between bins is not currently supported.
                    This variable overwrites the settings for 'ws_step', and instead allows 
                    the user to directly specify the binning properties, rather than deriving 
                    them from the data and an assigned step size. 
                    Defaults to None.
            
            N (:py:obj:`int`): Number of bootstrap evaluations for uncertainty quantification (UQ). 
                    If N = 1, no UQ will be performed. 
                    Defaults to 1.

            percentiles (:py:obj:`list[float]`): Confidence bounds for uncertainty quantification (UQ) in percents. 
                    This value is only relevant if N > 1 is specified. 
                    Defaults to [5.0, 95.0].

            balance_bins_between_dfs (:py:obj:`bool`): Balances the bins by the frequency of occurrence for each 
                    wind direction and wind speed bin in the collective of dataframes. The frequency of a
                    certain bin is equal to the minimum number of occurrences among all the dataframes. 
                    This ensures an "apples to apples" comparison. Recommended to set to 'True'. 
                    Will avoid bin rebalancing if the underlying 'wd' and 'ws' occurrences are identical 
                    between all dataframes (i.e. comparing SCADA data to FLORIS predictions of the same data).
                    Defaults to True.

            return_detailed_output (:py:obj:`bool`): Calculates and returns detailed energy ratio information useful for debugging
                    and evaluating flaws in the data. This can impact the speed of calculations but can be very useful. 
                    This information is written to 'self.df_lists[i]['er_results_info_dict']'. The dictionary contains 
                    two fields, 'df_per_wd_bin' and 'df_per_ws_bin'. 'df_per_wd_bin' provides an overview of the energy ratio for 
                    every wind direction bin, covering the collective effect of all wind speeds in the data. 'df_per_ws_bin' provides 
                    more information and displays the energy ratio for every wind direction and wind speed bin, among others. 
                    This is particularly helpful in determining if the bins are well balanced. 
                    Defaults to False.

            num_blocks (:py:obj:`int`): Number of blocks to use in block boostrapping. 
                    If num_blocks = -1, then do not use block bootstrapping and follow the 
                    normal approach of randomly sampling 'num_samples' with replacement.  
                    Defaults to -1.

            verbose (:py:obj:`bool`): Specify printing to console. 
                    Defaults to True.

        Returns:
            fi_tuned (:py:obj:`FlorisInterface`): Tuned FLORIS model.

            err (:py:obj:`float`): Mean squared error. Currently derived from energy ratios. # TODO: Will  eventually add capabilities for raw power and other evaluation metrics 

        """
        # Extract yaml dictionary from the untuned FLORIS model object for paramater tuning
        fi_dict_mod = self.fi_untuned.floris.as_dict()

        # Generate FLORIS dataframe for SCADA comparison
        df_floris = self.get_floris_df(fi=self.fi_untuned,
                                       pow_ref_columns=pow_ref_columns,
                                       time_series=time_series)

        # Calculate the mean squared error between SCADA and FLORIS energy ratios for each value in the range of parameters
        errs = []

        # If baseline case, tune wake expansion rate(s) 
        if case == 'baseline':
            for i in params:
                # Update 1st wake expansion rate parameter
                # TODO: Need to address breakpoints case
                fi_dict_mod['wake']['wake_velocity_parameters']['empirical_gauss']\
                ['wake_expansion_rates'][0] += i

                # Instantiate FLORIS model object with updated wake expansion rate parameter
                self.fi_tuned = FlorisInterface(fi_dict_mod)

                # Generate FLORIS dataframe for SCADA comparison
                df_floris = self.get_floris_df(fi=self.fi_tuned,
                                               pow_ref_columns=pow_ref_columns,
                                               time_series=time_series)
                
                # Calculate error
                err = self.evaluate_error(case=case,
                                          df_floris=df_floris, 
                                          wd_step=wd_step,
                                          ws_step=ws_step,
                                          wd_bin_width=wd_bin_width,
                                          wd_bins=wd_bins,
                                          ws_bins=ws_bins,
                                          N=N,
                                          percentiles=percentiles,
                                          balance_bins_between_dfs=balance_bins_between_dfs,
                                          return_detailed_output=return_detailed_output,
                                          num_blocks=num_blocks,
                                          verbose=verbose)
                
                # Track error and parameter value
                errs.append(err)
                
        # If controlled case, tune horizontal deflection gain
        elif case == 'controlled':
            for i in params:
                # Update horizontal deflection gain parameter
                fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
                ['horizontal_deflection_gain_D'] = i

                # Instantiate FLORIS model object with updated horizontal deflection gain parameter
                self.fi_tuned = FlorisInterface(fi_dict_mod)

                # Generate FLORIS dataframe for SCADA comparison
                df_floris = self.get_floris_df(fi=self.fi_tuned,
                                               pow_ref_columns=pow_ref_columns,
                                               time_series=time_series)
                
                # Calculate error
                err = self.evaluate_error(case=case,
                                          df_floris=df_floris, 
                                          wd_step=wd_step,
                                          ws_step=ws_step,
                                          wd_bin_width=wd_bin_width,
                                          wd_bins=wd_bins,
                                          ws_bins=ws_bins,
                                          N=N,
                                          percentiles=percentiles,
                                          balance_bins_between_dfs=balance_bins_between_dfs,
                                          return_detailed_output=return_detailed_output,
                                          num_blocks=num_blocks,
                                          verbose=verbose)
                
                # Track error
                errs.append(err)

        else:
            raise ValueError("Can only evaluate the 'baseline' or 'controlled' case.")
        
        # Return tuned FLORIS model and associated error
        # return self.fi_tuned, err
        return errs

    def write_yaml(self, filepath: str):
        """
        Write tuned FLORIS parameters to a YAML file.

        Args:
            filepath (:py:obj:`str`): Path that YAML file will be written to. 
         
        """

        # Check if file already exists
        if os.path.isfile(filepath):
             print(f'FLORIS YAML file {filepath} exists. Skipping...')

        # If it does not exist, write a new YAML file for tuned FLORIS parameters
        else:         
             fi_dict = self.fi_tuned.floris.as_dict()

             # Save the file path for future reference
             self.yaml = filepath
             
             print(f'Writing new FLORIS YAML file to `{filepath}`...')
             
             # Wrtie the YAML file
             with open(filepath, 'w') as f:
                yaml.dump(fi_dict, f)
                
             print('Finished writing FLORIS YAML file.')

    def get_untuned_floris(self):
        """
        Return untuned FLORIS model.

        Returns:
            fi_untuned (:py:obj:`FlorisInterface`): Untuned FLORIS model.
        
        """

        return self.fi_untuned

    def get_tuned_floris(self):
        """
        Return tuned FLORIS model.

        Returns:
            fi_tuned (:py:obj:`FlorisInterface`): Tuned FLORIS model.

        """

        return self.fi_tuned

    def get_yaml(self):
        """
        Return directory of the YAML file containing the tuned FLORIS parameters.

        Returns:
            yaml (:py:obj:`str`): Directory of YAML file.

        """

        return self.yaml

    



