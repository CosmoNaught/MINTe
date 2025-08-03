# minte/data/module.py
"""Data module for handling all data-related operations."""

import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .dataset import TimeSeriesDataset, collate_fn

logger = logging.getLogger(__name__)


class DataModule:
    """Module to handle all data-related operations."""
    
    def __init__(self, config: Config):
        """Initialize the data module with configuration."""
        self.config = config
        self.static_covars = [
            "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
            "seasonal", "routine", "itn_use", "irs_use",
            "itn_future", "irs_future", "lsm"
        ]
        self.df = None
        self.static_scaler = None
        self.train_param_sims = None
        self.val_param_sims = None
        self.test_param_sims = None
        self.train_params = None
        self.val_params = None
        self.test_params = None
        self.input_size = None
        self.target_column = "prevalence" if self.config.predictor == "prevalence" else "cases"
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from DuckDB database with performance optimizations."""
        logger.info(f"Connecting to DuckDB and fetching {self.config.predictor} data from {self.config.db_path}")
        total_start_time = time.time()

        # Connection with optimized settings
        con = duckdb.connect(self.config.db_path, read_only=True)
        con.execute("PRAGMA memory_limit='24GB';")
        con.execute("PRAGMA threads=16;")

        param_where_clause = ""
        if self.config.param_limit != "all":
            param_where_clause = f"WHERE parameter_index < {self.config.param_limit}"

        distinct_sims_subquery = f"""
            SELECT DISTINCT parameter_index, simulation_index, global_index
            FROM {self.config.table_name}
            {param_where_clause}
        """

        if self.config.sim_limit != "all":
            random_sims_subquery = f"""
                SELECT parameter_index, simulation_index, global_index
                FROM (
                    SELECT
                        parameter_index,
                        simulation_index,
                        global_index,
                        ROW_NUMBER() OVER (
                            PARTITION BY parameter_index
                            ORDER BY RANDOM()
                        ) AS rn
                    FROM ({distinct_sims_subquery})
                )
                WHERE rn <= {self.config.sim_limit}
            """
        else:
            random_sims_subquery = distinct_sims_subquery

        # Different handling for prevalence vs cases
        if self.config.predictor == "prevalence":
            final_query = self._build_prevalence_query(random_sims_subquery)
        else:
            final_query = self._build_cases_query(random_sims_subquery)

        df = con.execute(final_query).df()
        con.close()

        query_time = time.time() - total_start_time
        logger.info(f"Data fetched in {query_time:.2f} seconds.")
        
        return df
        
    def _build_prevalence_query(self, random_sims_subquery: str) -> str:
        """Build query for prevalence data."""
        cte_subquery = f"""
            SELECT
                t.parameter_index,
                t.simulation_index,
                t.global_index,
                t.timesteps,
                CASE WHEN t.n_age_0_1825 = 0 THEN NULL
                     ELSE CAST(t.n_detect_lm_0_1825 AS DOUBLE) / t.n_age_0_1825
                END AS raw_prevalence,
                t.eir,
                t.dn0_use,
                t.dn0_future,
                t.Q0,
                t.phi_bednets,
                t.seasonal,
                t.routine,
                t.itn_use,
                t.irs_use,
                t.itn_future,
                t.irs_future,
                t.lsm
            FROM {self.config.table_name} t
            JOIN ({random_sims_subquery}) rs
            USING (parameter_index, simulation_index)
        """

        preceding = self.config.window_size - 1
        last_6_years_day = 6 * 365

        return f"""
            WITH cte AS (
                {cte_subquery}
            )
            SELECT
                parameter_index,
                simulation_index,
                global_index,
                ROW_NUMBER() OVER (
                    PARTITION BY parameter_index, simulation_index
                    ORDER BY timesteps
                ) AS timesteps,
                AVG(raw_prevalence) OVER (
                    PARTITION BY parameter_index, simulation_index
                    ORDER BY timesteps
                    ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
                ) AS prevalence,
                eir,
                dn0_use,
                dn0_future,
                Q0,
                phi_bednets,
                seasonal,
                routine,
                itn_use,
                irs_use,
                itn_future,
                irs_future,
                lsm
            FROM cte
            WHERE cte.timesteps >= {last_6_years_day}
              AND (cte.timesteps % {self.config.window_size}) = 0
            ORDER BY parameter_index, simulation_index, timesteps
        """
        
    def _build_cases_query(self, random_sims_subquery: str) -> str:
        """Build query for cases data."""
        cte_subquery = f"""
            SELECT
                t.parameter_index,
                t.simulation_index,
                t.global_index,
                t.timesteps,
                t.n_inc_clinical_0_36500,
                t.n_age_0_36500,
                CASE WHEN t.n_age_0_36500 = 0 THEN NULL
                    ELSE 1000.0 * CAST(t.n_inc_clinical_0_36500 AS DOUBLE) / t.n_age_0_36500
                END AS raw_cases,
                t.eir,
                t.dn0_use,
                t.dn0_future,
                t.Q0,
                t.phi_bednets,
                t.seasonal,
                t.routine,
                t.itn_use,
                t.irs_use,
                t.itn_future,
                t.irs_future,
                t.lsm
            FROM {self.config.table_name} t
            JOIN ({random_sims_subquery}) rs
            USING (parameter_index, simulation_index)
        """

        last_6_years_day = 6 * 365
        aggregation_window = self.config.window_size

        return f"""
            WITH cte AS (
                {cte_subquery}
            ),
            timestep_groups AS (
                SELECT
                    parameter_index,
                    simulation_index, 
                    global_index,
                    FLOOR((timesteps - {last_6_years_day}) / {aggregation_window}) AS group_id,
                    1000.0 * SUM(n_inc_clinical_0_36500) / SUM(n_age_0_36500) AS cases,
                    MAX(eir) AS eir,
                    MAX(dn0_use) AS dn0_use,
                    MAX(dn0_future) AS dn0_future,
                    MAX(Q0) AS Q0,
                    MAX(phi_bednets) AS phi_bednets,
                    MAX(seasonal) AS seasonal,
                    MAX(routine) AS routine,
                    MAX(itn_use) AS itn_use,
                    MAX(irs_use) AS irs_use,
                    MAX(itn_future) AS itn_future,
                    MAX(irs_future) AS irs_future,
                    MAX(lsm) AS lsm
                    FROM cte
                    WHERE timesteps >= {last_6_years_day}
                    GROUP BY parameter_index, simulation_index, global_index, group_id
            )
            SELECT
                parameter_index,
                simulation_index,
                global_index,
                ROW_NUMBER() OVER (
                    PARTITION BY parameter_index, simulation_index
                    ORDER BY group_id
                ) AS timesteps,
                cases,
                eir,
                dn0_use,
                dn0_future,
                Q0,
                phi_bednets,
                seasonal,
                routine,
                itn_use,
                irs_use,
                itn_future,
                irs_future,
                lsm
            FROM timestep_groups
            ORDER BY parameter_index, simulation_index, group_id
        """
        
    def prepare_data(self) -> None:
        """Prepare data for model training."""
        # 1. Fetch data
        self.df = self.fetch_data()
        
        # 2. Filter data based on threshold
        self._filter_by_threshold()
        
        # 3. Create train/val/test split
        self._create_data_split()
        
        # 4. Create and fit feature scalers
        self._create_scalers()
        
        # 5. Determine input size for models
        self._set_input_size()
        
        logger.info("Data preparation completed successfully.")
        
    def _filter_by_threshold(self) -> None:
        """Filter data based on threshold appropriate for the predictor type."""
        group_cols = ["parameter_index", "simulation_index"]
        
        if self.config.predictor == "prevalence":
            group_means = self.df.groupby(group_cols)["prevalence"].mean().reset_index()
            valid_groups = group_means[group_means["prevalence"] >= self.config.min_prevalence]
            logger.info(f"Filtered data to {len(valid_groups)} parameter-simulation pairs with prevalence >= {self.config.min_prevalence}")
        else:
            group_means = self.df.groupby(group_cols)["cases"].mean().reset_index()
            valid_groups = group_means[group_means["cases"] > 0]
            logger.info(f"Filtered data to {len(valid_groups)} parameter-simulation pairs with positive case counts")
        
        valid_keys = set(zip(valid_groups["parameter_index"], valid_groups["simulation_index"]))
        self.df["param_sim"] = list(zip(self.df["parameter_index"], self.df["simulation_index"]))
        self.df = self.df[self.df["param_sim"].isin(valid_keys)]
        
    def _create_data_split(self) -> None:
        """Create or load train/validation/test split."""
        param_sim_to_global = {}
        for _, row in self.df.iterrows():
            param_idx = row["parameter_index"]
            sim_idx = row["simulation_index"]
            global_idx = row["global_index"]
            param_sim_to_global[(param_idx, sim_idx)] = global_idx
        
        if self.config.use_existing_split and os.path.exists(self.config.split_file):
            logger.info(f"Loading existing train/val/test split from {self.config.split_file}")
            self._load_existing_split()
        else:
            logger.info("Creating new train/val/test split (70/15/15)")
            self._create_new_split(param_sim_to_global)
            
        logger.info(f"Data split: {len(self.train_param_sims)} train, {len(self.val_param_sims)} validation, {len(self.test_param_sims)} test parameter-simulation pairs")
        logger.info(f"Number of unique parameters: {len(self.train_params)} train, {len(self.val_params)} validation, {len(self.test_params)} test")
        
    def _load_existing_split(self) -> None:
        """Load existing train/validation/test split."""
        split_df = pd.read_csv(self.config.split_file)
        
        self.train_params = set(split_df[split_df["split"] == "train"]["parameter_index"])
        self.val_params = set(split_df[split_df["split"] == "validate"]["parameter_index"])
        self.test_params = set(split_df[split_df["split"] == "test"]["parameter_index"])
        
        self.train_param_sims = set()
        self.val_param_sims = set()
        self.test_param_sims = set()
        
        for _, row in split_df.iterrows():
            param_idx = row["parameter_index"]
            sim_idx = row["simulation_index"]
            split = row["split"]
            
            if split == "train":
                self.train_param_sims.add((param_idx, sim_idx))
            elif split == "validate":
                self.val_param_sims.add((param_idx, sim_idx))
            elif split == "test":
                self.test_param_sims.add((param_idx, sim_idx))
                
    def _create_new_split(self, param_sim_to_global: Dict) -> None:
        """Create new train/validation/test split."""
        param_sim_groups = self.df.groupby(["parameter_index", "simulation_index"])
        all_param_sims = list(param_sim_groups.groups.keys())
        
        unique_parameters = list(set([ps[0] for ps in all_param_sims]))
        random.shuffle(unique_parameters)

        n_params = len(unique_parameters)
        n_train_params = int(0.7 * n_params)
        n_val_params = int(0.15 * n_params)

        self.train_params = set(unique_parameters[:n_train_params])
        self.val_params = set(unique_parameters[n_train_params : n_train_params + n_val_params])
        self.test_params = set(unique_parameters[n_train_params + n_val_params :])

        self.train_param_sims = set(ps for ps in all_param_sims if ps[0] in self.train_params)
        self.val_param_sims = set(ps for ps in all_param_sims if ps[0] in self.val_params)
        self.test_param_sims = set(ps for ps in all_param_sims if ps[0] in self.test_params)

        split_info = []
        for ps in all_param_sims:
            param_idx, sim_idx = ps
            global_idx = param_sim_to_global.get((param_idx, sim_idx), None)
            
            if param_idx in self.train_params:
                split = 'train'
            elif param_idx in self.val_params:
                split = 'validate'
            else:
                split = 'test'
            
            split_info.append({
                'parameter_index': param_idx,
                'simulation_index': sim_idx,
                'global_index': global_idx,
                'split': split
            })

        split_df = pd.DataFrame(split_info)
        split_df.to_csv(self.config.split_file, index=False)
        logger.info(f"Saved train/validation/test split information to {self.config.split_file}")
        
    def _create_scalers(self) -> None:
        """Create and fit feature scalers."""
        static_scalar_values = self.df[self.static_covars].values.astype(np.float32)
        self.static_scaler = StandardScaler()
        self.static_scaler.fit(static_scalar_values)
        
        scaler_path = os.path.join(self.config.output_dir, "static_scaler.pkl")
        pd.to_pickle(self.static_scaler, scaler_path)
        logger.info(f"Feature scaler saved to {scaler_path}")
        
    def _set_input_size(self) -> None:
        """Determine input size for models based on feature encoding."""
        if self.config.use_cyclical_time:
            self.input_size = 2 + len(self.static_covars)
        else:
            self.input_size = 1 + len(self.static_covars)
        logger.info(f"Input size for models set to {self.input_size}")
            
    def build_data_list(self, param_sims: set) -> List[Dict]:
        """Build normalized data list with vectorized operations."""
        param_sim_groups = self.df.groupby(["parameter_index", "simulation_index"])
        data_list = []
        
        for ps in param_sims:
            subdf = param_sim_groups.get_group(ps).sort_values("timesteps")
            T = len(subdf)

            static_vals = subdf.iloc[0][self.static_covars].values.astype(np.float32)
            static_vals = self.static_scaler.transform(static_vals.reshape(1, -1)).flatten()

            t = subdf["timesteps"].values.astype(np.float32)

            if self.config.use_cyclical_time:
                if self.config.predictor == "cases":
                    day_of_year = (t * self.config.window_size) % 365.0
                else:
                    day_of_year = t % 365.0
                sin_t = np.sin(2 * math.pi * day_of_year / 365.0)
                cos_t = np.cos(2 * math.pi * day_of_year / 365.0)
                
                X = np.zeros((T, 2 + len(self.static_covars)), dtype=np.float32)
                X[:, 0] = sin_t
                X[:, 1] = cos_t
                X[:, 2:] = np.tile(static_vals, (T, 1))
            else:
                t_min, t_max = np.min(t), np.max(t)
                t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
                
                X = np.zeros((T, 1 + len(self.static_covars)), dtype=np.float32)
                X[:, 0] = t_norm
                X[:, 1:] = np.tile(static_vals, (T, 1))

            Y = subdf[self.target_column].values.astype(np.float32)

            # Get target variable based on predictor type
            if self.config.predictor == "prevalence":
                Y = subdf["prevalence"].values.astype(np.float32)
            else:  # cases
                Y = subdf["cases"].values.astype(np.float32)
                # Log transform cases to handle skewness
                Y = np.log1p(Y)  # log(1 + cases)

            data_list.append({
                "time_series": X,  
                "targets": Y,        
                "length": T,
                "param_sim_id": ps
            })
        return data_list

        
    def create_datasets(self, lookback: int) -> Dict[str, Dataset]:
        """Create datasets for train, validation, and test sets."""
        train_groups = self.build_data_list(self.train_param_sims)
        val_groups = self.build_data_list(self.val_param_sims)
        test_groups = self.build_data_list(self.test_param_sims)

        train_dataset = TimeSeriesDataset(train_groups, lookback=lookback)
        val_dataset = TimeSeriesDataset(val_groups, lookback=lookback)
        test_dataset = TimeSeriesDataset(test_groups, lookback=lookback)
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        
    def create_dataloaders(self, datasets: Dict[str, Dataset], batch_size: int) -> Dict[str, DataLoader]:
        """Create optimized data loaders from datasets."""
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,   
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=8,
            drop_last=True
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=8
        )
        
        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=8
        )
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }