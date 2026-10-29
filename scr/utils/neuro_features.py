import typing as tp

import pydantic

import neuralset as ns


neuro_devices = tp.Union[ns.features.neuro.Meg | ns.features.neuro.Eeg]

Neuro_devices = tp.Annotated[neuro_devices, pydantic.Field(discriminator="name")]
