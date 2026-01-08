import os
import json
import logging

from model_setup.metrics.mech_interp.identify_neurons import Entropy_Neurons_Identification
from model_setup.metrics.mech_interp.neuron_activation_recorder import NeuronActivationRecorder

logger = logging.getLogger(__name__)

class MechInterp:

    @classmethod
    def get_recorder(cls, model_name, model, mech_interp_ident_methods, entropy_neurons):
        path_to_identified_neurons = os.path.join(
            "./unified_pipeline/model_setup/metrics/mech_interp",
            model_name,
            f"{"_".join(mech_interp_ident_methods)}_{entropy_neurons}_neurons.json"
        )
        logger.info(f"Initializing MechInterp for model: {model_name}")
        methods_str = "_".join(mech_interp_ident_methods)
        base_dir = os.path.join("./unified_pipeline/models/metrics/mech_interp", model_name)

        if not os.path.exists(base_dir):
            logger.info(f"Directory {base_dir} does not exist. Creating it.")
            os.makedirs(base_dir, exist_ok=True)
        
        path_to_identified_neurons = os.path.join(
            base_dir,
            f"{methods_str}_{entropy_neurons}_neurons.json"
        )

        if not os.path.isfile(path_to_identified_neurons):
            logger.info(f"Identified neurons file not found at {path_to_identified_neurons}. Starting identification process...")
            identify_neurons = Entropy_Neurons_Identification(
                model_name=model_name,
                model=model,
                output_path=path_to_identified_neurons,
                k=entropy_neurons
            )
            max_dev = "max_dev" in mech_interp_ident_methods
            var = "var" in mech_interp_ident_methods
            cos_sim = "cos_sim" in mech_interp_ident_methods

            identify_neurons.identify(
                identify_by_cosine_sim=cos_sim,
                identify_by_max_std_dev=max_dev,
                identify_by_variance=var,
                file_name=path_to_identified_neurons)
            logger.info("Neuron identification complete.")
        else:
            logger.info(f"Loading cached identified neurons from: {path_to_identified_neurons}")
            
        try:
            with open(path_to_identified_neurons, 'r') as f:
                neuron_indices = json.load(f)
            logger.info(f"Loaded {len(neuron_indices)} neuron indices.")
        except Exception as e:
            logger.error(f"Failed to read neuron file {path_to_identified_neurons}: {e}")
            raise e
        
        if model_name == "unsloth/Meta-Llama-3.1-8B":
            layer_name = "model.layers.31.mlp.down_proj"
        else:
            error_msg = f"MechInterp not configured for model: {model_name}. Please define 'layer_name'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Attaching NeuronActivationRecorder to layer: {layer_name}")
        return NeuronActivationRecorder(model, neuron_indices, layer_name=layer_name), neuron_indices



