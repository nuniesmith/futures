"""
Prediction Model Manager

This module handles the creation, loading, and management of
prediction models for different assets.
"""

import os

from lib.model._shims import logger

# --- Stubs for modules not present in this project ---
AssetDataManager = None  # stub: data.manager.AssetDataManager

try:
    from lib.model.statistical.bayesian import BayesianLinearRegression as BayesianPredictor
except ImportError:
    BayesianPredictor = None  # type: ignore[assignment,misc]

try:
    from lib.model.ml.gaussian import GaussianModel as GaussianPredictor
except ImportError:
    GaussianPredictor = None  # type: ignore[assignment,misc]

try:
    from lib.model.ml.polynomial import PolynomialRegression as PolynomialPredictor
except ImportError:
    PolynomialPredictor = None  # type: ignore[assignment,misc]

get_config = None  # stub: core.constants.manager.get_config


class ModelManager:
    """
    Manages prediction models for multiple assets, including
    creation, initialization, and persistence.
    """

    def __init__(self, data_fetcher: AssetDataManager | None = None, model_dir: str | None = None):
        """
        Initialize the model manager.

        Args:
            data_fetcher: Data fetcher instance for retrieving market data
            model_dir: Directory where models are stored
        """
        # Get constants
        self.constants = get_config()

        self.data_fetcher = data_fetcher or AssetDataManager()
        self.model_dir = model_dir or self.constants.MODEL_DIR
        self.models = {}
        self.model_types = {"gold": "bayesian", "bitcoin": "gaussian"}

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize prediction models for each asset."""
        logger.info("Initializing prediction models")

        for asset in self.constants.SUPPORTED_ASSETS:
            model_type = self.model_types.get(asset, "bayesian")
            self.models[asset] = self._create_model(asset, model_type)

    def _create_model(self, asset: str, model_type: str) -> BayesianPredictor | GaussianPredictor | PolynomialPredictor:
        """
        Create a predictor model of the specified type.

        Args:
            asset: Asset the model will predict
            model_type: Type of prediction model to create

        Returns:
            Initialized predictor model
        """
        valid_model_types = getattr(self.constants, "VALID_MODEL_TYPES", ["bayesian", "gaussian", "polynomial"])
        if model_type not in valid_model_types:
            logger.warning(f"Invalid model type '{model_type}', defaulting to bayesian")
            model_type = "bayesian"

        try:
            # Create model based on type
            if model_type == "bayesian":
                # Get model parameters
                params = getattr(self.constants, "MODEL_PARAMS", {}).get("bayesian", {})
                return BayesianPredictor(asset=asset, data_fetcher=self.data_fetcher, **params)

            elif model_type == "gaussian":
                # Get base Gaussian parameters
                gaussian_params = getattr(self.constants, "GAUSSIAN_PARAMS", {}).copy()

                # Get asset-specific Gaussian parameters and update base params
                asset_specific_params = {}
                if hasattr(self.constants, "get_asset_gaussian_params"):
                    asset_specific_params = self.constants.get_asset_gaussian_params(asset)
                gaussian_params.update(asset_specific_params)

                # Create the model with the combined parameters
                logger.debug(f"Creating Gaussian model for {asset} with parameters: {gaussian_params}")
                return GaussianPredictor(asset=asset, data_fetcher=self.data_fetcher, **gaussian_params)

            elif model_type == "polynomial":
                # Get model parameters
                params = getattr(self.constants, "MODEL_PARAMS", {}).get("polynomial", {})
                return PolynomialPredictor(asset=asset, data_fetcher=self.data_fetcher, **params)

            else:
                # Default
                logger.warning(f"Unrecognized model type '{model_type}', using bayesian as fallback")
                return BayesianPredictor(asset=asset, data_fetcher=self.data_fetcher)

        except Exception as e:
            logger.error(f"Error creating {model_type} model for {asset}: {str(e)}")
            # Return a default model as fallback
            return BayesianPredictor(asset=asset, data_fetcher=self.data_fetcher)

    def get_model_path(self, asset: str) -> str:
        """
        Get the file path for a model.

        Args:
            asset: Asset type (e.g., 'gold', 'bitcoin')

        Returns:
            Full path to the model file
        """
        model_type = self.model_types.get(asset, "bayesian")
        return os.path.join(self.model_dir, f"{asset}_{model_type}_model.pkl")

    async def update_models(
        self, asset_model_map: dict[str, str] | None = None, load_existing: bool = True
    ) -> dict[str, bool]:
        """
        Update the model types for specified assets.

        Args:
            asset_model_map: Mapping from asset names to model types
            load_existing: Whether to load existing models after update

        Returns:
            Dictionary indicating which assets were updated successfully
        """
        if asset_model_map is None:
            asset_model_map = {}

        logger.info(f"Updating models: {asset_model_map}")

        results = {}
        for asset, model_type in asset_model_map.items():
            if asset not in self.constants.SUPPORTED_ASSETS:
                logger.warning(f"Skipping unsupported asset: {asset}")
                results[asset] = False
                continue

            # Only update if the model type has changed
            if model_type != self.model_types.get(asset):
                try:
                    self.model_types[asset] = model_type
                    self.models[asset] = self._create_model(asset, model_type)
                    logger.info(f"Created new {model_type} model for {asset}")
                    results[asset] = True
                except Exception as e:
                    logger.error(f"Failed to update model for {asset}: {e}")
                    results[asset] = False
            else:
                results[asset] = True  # No change needed

        # Load existing trained models if requested
        if load_existing:
            await self.load_existing_models()

        return results

    async def load_existing_models(self) -> dict[str, bool]:
        """
        Load existing trained models.

        Returns:
            Dictionary indicating which models were loaded successfully
        """
        results = {}
        for asset in self.constants.SUPPORTED_ASSETS:
            model_path = self.get_model_path(asset)
            if os.path.exists(model_path):
                try:
                    self.models[asset].load(model_path)
                    logger.info(f"Loaded existing model for {asset} from {model_path}")
                    results[asset] = True
                except Exception as e:
                    logger.error(f"Error loading model for {asset}: {str(e)}")
                    results[asset] = False
            else:
                logger.warning(f"No model file found for {asset} at {model_path}")
                results[asset] = False

        return results

    def get_model(self, asset: str) -> BayesianPredictor | GaussianPredictor | PolynomialPredictor | None:
        """
        Get the model for a specific asset.

        Args:
            asset: Asset name

        Returns:
            Model instance or None if not available
        """
        return self.models.get(asset)

    def get_model_type(self, asset: str) -> str:
        """
        Get the model type for a specific asset.

        Args:
            asset: Asset name

        Returns:
            Model type as string
        """
        return self.model_types.get(asset, "bayesian")
