# minte/__init__.py
"""
Malaria Forecast Package

A comprehensive package for time series forecasting of malaria prevalence 
and clinical cases using RNN models.
"""

__version__ = "0.1.0"
__author__ = "Cosmo Santoni"
__email__ = "cosmo.santoni@imperial.ac.uk"

from .api import MalariaForecast, run_forecast
from .config import Config

__all__ = ["MalariaForecast", "run_forecast", "Config"]