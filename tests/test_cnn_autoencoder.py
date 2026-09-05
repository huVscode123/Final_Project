"""Unit tests for CNN Autoencoder checkpoint compatibility."""

from copy import deepcopy

import torch

from core.cnn_autoencoder import CNNAutoencoder, load_compatible_state_dict


def test_legacy_decoder_output_layer_loads_strictly():
    """A pre-Upsample ConvTranspose output layer must load into Conv2d."""
    source_model = CNNAutoencoder(latent_dim=32)
    expected_weight = source_model.state_dict()['decoder.deconv_layers.10.weight']
    expected_bias = source_model.state_dict()['decoder.deconv_layers.10.bias']

    legacy_state = deepcopy(source_model.state_dict())
    del legacy_state['decoder.deconv_layers.10.weight']
    del legacy_state['decoder.deconv_layers.10.bias']
    # Reverse the compatibility conversion to construct a real legacy layout.
    legacy_state['decoder.deconv_layers.9.weight'] = expected_weight.flip(
        2, 3
    ).permute(1, 0, 2, 3).contiguous()
    legacy_state['decoder.deconv_layers.9.bias'] = expected_bias.clone()

    loaded_model = CNNAutoencoder(latent_dim=32)
    load_compatible_state_dict(loaded_model, legacy_state)

    loaded_state = loaded_model.state_dict()
    assert torch.equal(
        loaded_state['decoder.deconv_layers.10.weight'], expected_weight
    )
    assert torch.equal(
        loaded_state['decoder.deconv_layers.10.bias'], expected_bias
    )
