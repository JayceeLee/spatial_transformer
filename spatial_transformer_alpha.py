import tensorflow as tf


def transformer(inputs, theta, output_shape, sampling_kernel='bilinear'):
    """Affine spatial transformation of inputs according to theta and the output shape

    Args:
        inputs: tensor with the shape of n * h* w * c
        theta: affine transformation parameters with the shape of num_batch * transform_params
        output_shape: the output shape with the form [height, width]
        sampling_kernel: the interpolation methods. 1.bilinear, 2.kronecker, default: bilinear

    Returns:
        outputs: spatial transformed inputs tensor
    """
    input_shape = inputs.get_shape().as_list()
    num_batch = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]
    num_channel = input_shape[3]
    output_height = output_shape[0]
    output_width = output_shape[1]

    x_grid, y_grid = _output_meshgrid(output_height, output_width)

    theta = _theta_filter(theta, num_batch)

    input_x_coordinates, input_y_coordinates = _input_sampling_coordinates(x_grid, y_grid, theta,
                                                                           input_height, input_width)

    if sampling_kernel == 'bilinear':
        outputs = _bilinear_interpolation(inputs, input_x_coordinates, input_y_coordinates, num_batch, input_height,
                                          input_width, num_channel, output_height, output_width)
    elif sampling_kernel == 'kronecker':
        outputs = _integer_interpolation(inputs, input_x_coordinates, input_y_coordinates, num_batch, input_height,
                                         input_width, num_channel, output_height, output_width)
    else:
        outputs = _bilinear_interpolation(inputs, input_x_coordinates, input_y_coordinates, num_batch, input_height,
                                          input_width, num_channel, output_height, output_width)

    return outputs


def _theta_filter(theta, num_batch):
    """
    Experiments: shear the spatial transformation parameter to reduce the black boundary
    """
    # corner points to create the attention bounding box
    # input theta should have the form of [s_x, t_x, s_y, t_y]

    # convert to the form of [-s_x + t_x, s_x + t_x, -s_y + t_y, s_y + t_y]
    theta_2_corners = tf.constant([[-1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [0, 0, -1, 1],
                                   [0, 0, 1, 1]], tf.float32)
    corner_coordinates = tf.matmul(theta, theta_2_corners)

    corner_coordinates_new = tf.clip_by_value(corner_coordinates, -1.0, 1.0)
    corners_2_theta = tf.constant([[-0.5, 0.5, .0, .0],
                                   [0.5, 0.5, .0, .0],
                                   [.0, .0, -0.5, .5],
                                   [.0, .0, 0.5, 0.5]], tf.float32)
    theta_new = tf.matmul(corner_coordinates_new, corners_2_theta)

    s_x_new, t_x_new, s_y_new, t_y_new = tf.split(theta_new, 4, axis=1)
    aux_zeros = tf.zeros([num_batch, 1])

    # output theta should have the form: [[s_x, 0, t_x], [0, s_y, t_y]]
    theta_new = tf.concat([s_x_new, aux_zeros, t_x_new, aux_zeros, s_y_new, t_y_new], axis=1)
    theta_format = tf.reshape(theta_new, [num_batch, 2, 3])

    return theta_format


def _output_meshgrid(height, width):
    """
    To generate the normalized meshgrid of the output image according to its size.

    Args:
        size of the target output image

    Returns:
        the meshgrid along vertical (x_grid) and horizon (y_grid)
    """
    height_norm = tf.linspace(-1.0, 1.0, height)
    width_norm = tf.linspace(-1.0, 1.0, width)
    y_grid, x_grid = tf.meshgrid(width_norm, height_norm)

    return x_grid, y_grid


def _input_sampling_coordinates(x_grid, y_grid, theta, height, width):
    """
    Mapping from outputs coordinates to inputs coordinates according to theta and the size of the inputs
    Returns: num_batch*out_height*out_weight
    """
    num_batch = theta.get_shape().as_list()[0]
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    theta_f = tf.cast(theta, 'float32')

    x_grid_flat = tf.reshape(x_grid, [-1])
    y_grid_flat = tf.reshape(y_grid, [-1])
    ones = tf.ones_like(x_grid_flat)
    output_coordinates = tf.stack([x_grid_flat, y_grid_flat, ones], axis=0)
    output_coordinates = tf.tile(tf.expand_dims(output_coordinates, 0), [num_batch, 1, 1])

    input_coordinates = tf.matmul(theta_f, output_coordinates)
    input_coordinates_clip = tf.clip_by_value(input_coordinates, -1.0, 1.0)
    input_x_coordinates, input_y_coordinates = tf.split(input_coordinates_clip, 2, axis=1)
    input_x_coordinates = (input_x_coordinates + 1.0) / 2.0 * (height_f - 1)
    input_y_coordinates = (input_y_coordinates + 1.0) / 2.0 * (width_f - 1)

    return tf.reshape(input_x_coordinates, [-1]), tf.reshape(input_y_coordinates, [-1])


def _bilinear_interpolation(inputs, x_coordinates, y_coordinates, num_batches, input_height, input_width, num_channels,
                            output_height, output_width):
    """
    Bilinear interpolation: 1. interpolated value calculated by the nearest four points at the square
                               in which the target points [x_coordinates, y_coordinates] locate
                            2. generate the spatial transformed outputs
    """
    # size: num_batch*output_height*output*width
    x_0_idx = tf.cast(tf.floor(x_coordinates), 'int32')
    y_0_idx = tf.cast(tf.floor(y_coordinates), 'int32')
    x_1_idx = tf.clip_by_value(x_0_idx + 1, 0, input_height - 1)
    y_1_idx = tf.clip_by_value(y_0_idx + 1, 0, input_width - 1)

    x_0_f = tf.cast(x_0_idx, 'float32')
    y_0_f = tf.cast(y_0_idx, 'float32')
    x_1_f = tf.cast(x_1_idx, 'float32')
    y_1_f = tf.cast(y_1_idx, 'float32')

    weight_00_flat = (x_1_f - x_coordinates) * (y_1_f - y_coordinates)
    weight_01_flat = (x_1_f - x_coordinates) * (y_coordinates - y_0_f)
    weight_10_flat = (x_coordinates - x_0_f) * (y_1_f - y_coordinates)
    weight_11_flat = (x_coordinates - x_0_f) * (y_coordinates - y_0_f)

    weight_00 = tf.reshape(weight_00_flat, [num_batches, output_height, output_width, 1])
    weight_01 = tf.reshape(weight_01_flat, [num_batches, output_height, output_width, 1])
    weight_10 = tf.reshape(weight_10_flat, [num_batches, output_height, output_width, 1])
    weight_11 = tf.reshape(weight_11_flat, [num_batches, output_height, output_width, 1])

    interpolated_value_00 = _get_interpolation_value(inputs, x_0_idx, y_0_idx, num_batches, input_height, input_width,
                                                     num_channels, output_height, output_width)
    interpolated_value_01 = _get_interpolation_value(inputs, x_0_idx, y_1_idx, num_batches, input_height, input_width,
                                                     num_channels, output_height, output_width)
    interpolated_value_10 = _get_interpolation_value(inputs, x_1_idx, y_0_idx, num_batches, input_height, input_width,
                                                     num_channels, output_height, output_width)
    interpolated_value_11 = _get_interpolation_value(inputs, x_1_idx, y_1_idx, num_batches, input_height, input_width,
                                                     num_channels, output_height, output_width)

    outputs = weight_00 * interpolated_value_00 + weight_01 * interpolated_value_01 \
              + weight_10 * interpolated_value_10 + weight_11 * interpolated_value_11

    return outputs


def _integer_interpolation(inputs, x_coordinates, y_coordinates, num_batches, input_height, input_width, num_channels,
                           output_height, output_width):
    """
    Integer interpolation using the kronecker function
    which indicates that the nearest point around the target sampling point is selected
    """
    x_idx = tf.cast(tf.round(x_coordinates), 'int32')
    y_idx = tf.cast(tf.round(y_coordinates), 'int32')
    interpolated_value = _get_interpolation_value(inputs, x_idx, y_idx, num_batches, input_height, input_width,
                                                  num_channels, output_height, output_width)

    return interpolated_value


def _get_interpolation_value(inputs, x_idx, y_idx, num_batches, input_height, input_width, num_channels,
                             output_height, output_width):
    """
    Calculate the interpolated value of the target points of outputs
    """
    input_im_size = input_height * input_width
    output_im_size = output_height * output_width

    # shape: [num_batches, out_im_size], [0:num_batches-1]' * ones([1, out_im_size])
    batch_base = tf.range(num_batches) * input_im_size
    ones_out_im_size = tf.ones([1, output_im_size], 'int32')
    sampling_idx_mat = tf.matmul(tf.expand_dims(batch_base, 1), ones_out_im_size)
    sampling_idx_shift_base = tf.reshape(sampling_idx_mat, [-1])

    # input_size shift per batch
    sampling_idx = sampling_idx_shift_base + x_idx * input_width + y_idx
    inputs_flat = tf.reshape(inputs, [-1, num_channels])
    value_sampled = tf.gather(inputs_flat, sampling_idx)
    value_sampled_formatted = tf.reshape(value_sampled, [num_batches, output_height, output_width, num_channels])

    # the input image is uint8 format
    return tf.cast(value_sampled_formatted, 'float32')

