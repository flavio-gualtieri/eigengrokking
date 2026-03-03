EXPERIMENTS_SCHEMA = {
    "name": "experiments_payload",
    "schema": {
        "type": "object",
        "properties": {
            "experiments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "enum": ["MODULAR", "MNIST", "FashionMNIST"]},
                        "modulus": {"type": "integer"},
                        "optimization_steps": {"type": "integer"},
                        "train_points": {"type": "number"},
                        "batch_size": {"type": "integer"},
                        "weight_decay": {"type": "number"},
                        "initialization_scale": {"type": "number"},
                        "run_spectral": {"type": "boolean"},
                    },
                    "required": [
                        "dataset",
                        "optimization_steps",
                        "train_points",
                        "batch_size",
                        "weight_decay",
                        "initialization_scale",
                        "run_spectral",
                        "modulus"
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["experiments"],
        "additionalProperties": False,
    },
}