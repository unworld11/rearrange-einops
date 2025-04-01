from validators import Validator
from utils import check_extra_arguments, get_additional_args
from transformations import input_based_transformation, update_input_tokens_mapping, Output_Transformations

def rearrange(array, pattern, **kwargs):
    """
    Rearranges an array based on the einops-like pattern and additional arguments.

    Steps:
    1. Validate the input array and pattern.
    2. Process extra arguments and parentheses.
    3. Perform input-based transformations.
    4. Update input tokens mapping based on transformations.
    5. Apply output transformations.
    6. Return the final transformed array.
    """

    v = Validator(array, pattern, **kwargs)

    array, input_tokens_mapping, input_tokens_shape_mapping, output_tokens_mapping = v.validate_and_return()

    check_extra_arguments(input_tokens_mapping, **kwargs)
    all_args = get_additional_args(input_tokens_mapping, input_tokens_shape_mapping, **kwargs)

    transformed_array = input_based_transformation(
                                                array,
                                                input_tokens_mapping,
                                                input_tokens_shape_mapping,
                                                **all_args)

    updated_tokens_mapping = update_input_tokens_mapping(
        input_tokens_mapping,
        **all_args,
    )

    d = Output_Transformations(transformed_array, token_mapping=updated_tokens_mapping, output_mapping=output_tokens_mapping)
    return d.transform()