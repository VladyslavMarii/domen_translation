import tensorflow as tf
from config import IMAGE_CHANNELS, IMAGE_RESOLUTION
from raw_gen_model.raw_gen_config import PATCH_SIZE


def create_satellite_encoder(patch_size=PATCH_SIZE, input_shape=(IMAGE_RESOLUTION["height"], IMAGE_RESOLUTION["width"], IMAGE_CHANNELS)):
    """
    Creates an "encoder" function that slices the image into patches
    and flattens each patch (no learned weights).

    Args:
        patch_size: Size of each patch (H and W).
        input_shape: Shape of the input image (height, width, channels).

    Returns:
        encode_satellite_image: A function that converts images into flattened patch tokens.
    """
    # Precompute some values for convenience/validation.
    num_h = input_shape[0] // patch_size
    num_w = input_shape[1] // patch_size
    c = input_shape[2]
    patch_dim = patch_size * patch_size * c

    def encode_satellite_image(satellite_image):
        """
        Slices the image into (patch_size x patch_size) patches,
        then flattens each patch.

        Args:
            satellite_image: A 4D tensor of shape [batch_size, H, W, C].
                             If your data is unbatched [H, W, C], you should add a batch dim.

        Returns:
            A 3D tensor of shape [batch_size, num_patches, patch_dim],
            where num_patches = (H/patch_size) * (W/patch_size)
                  patch_dim   = patch_size * patch_size * channels
        """
        # satellite_image: shape (batch_size, H, W, C)
        batch_size = tf.shape(satellite_image)[0]

        # Reshape into (batch_size, num_h, patch_size, num_w, patch_size, C)
        x = tf.reshape(
            satellite_image,
            [batch_size, num_h, patch_size, num_w, patch_size, c]
        )

        # Transpose to group the patch dimensions in a standard (num_h, num_w) layout:
        # shape -> (batch_size, num_h, num_w, patch_size, patch_size, C)
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])

        # Flatten the spatial patch dimensions:
        # shape -> (batch_size, num_h, num_w, patch_dim)
        x = tf.reshape(x, [batch_size, num_h, num_w, patch_dim])

        # Flatten (num_h * num_w) patches into a single sequence dimension:
        # shape -> (batch_size, num_patches, patch_dim)
        x = tf.reshape(x, [batch_size, num_h * num_w, patch_dim])

        return x

    return encode_satellite_image

def create_satellite_decoder(patch_size=PATCH_SIZE, input_shape=(IMAGE_RESOLUTION["height"], IMAGE_RESOLUTION["width"], IMAGE_CHANNELS)):
    """
    Creates a "decoder" function that reshapes flattened patch tokens
    back into the original image shape (no learned weights).

    Args:
        patch_size: Size of each patch (H and W).
        input_shape: (height, width, channels) of the original image.

    Returns:
        decode_satellite_tokens: A function that converts flattened patch tokens back to an image.
    """
    # Precompute some values for convenience/validation.
    H, W, C = input_shape
    num_h = H // patch_size
    num_w = W // patch_size
    patch_dim = patch_size * patch_size * C

    def decode_satellite_tokens(tokens):
        """
        Reshapes the flattened patches back to the original image shape.

        Args:
            tokens: A 3D tensor of shape [batch_size, num_patches, patch_dim],
                    where patch_dim = patch_size * patch_size * C.

        Returns:
            reconstructed_image: A 4D tensor of shape [batch_size, H, W, C].
        """
        batch_size = tf.shape(tokens)[0]

        # Reshape from (batch_size, num_patches, patch_dim) to (batch_size, num_h, num_w, patch_dim)
        x = tf.reshape(tokens, [batch_size, num_h, num_w, patch_dim])

        # Unflatten each patch from shape (patch_dim) to (patch_size, patch_size, C)
        x = tf.reshape(x, [batch_size, num_h, num_w, patch_size, patch_size, C])

        # Transpose to get shape (batch_size, num_h, patch_size, num_w, patch_size, C)
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])

        # Finally, reshape back to the original image (batch_size, H, W, C)
        reconstructed_image = tf.reshape(x, [batch_size, H, W, C])
        return reconstructed_image

    return decode_satellite_tokens