from typing import Optional, Union, List
from transformers import BlipProcessor
from transformers.processing_utils import Unpack
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from transformers.utils import TensorType
from transformers.models.blip.processing_blip import BlipProcessorKwargs

class CaptionBlipProcessor(BlipProcessor):
    """
    Processor for the BLIP model.

    This processor combines an image processor and a tokenizer to prepare inputs for the BLIP model.
    It can handle both images and text inputs, and provides methods to process them accordingly.

    The processor can be used to **prepared** images and text for the BLIP model, which can then be used for tasks like
    image captioning or visual question answering.

    For more details on how to use this processor, refer to the documentation of the `BlipProcessor` class.
    """    
    def __call__(
        self,
        images: TensorType = None,
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[BlipProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        text_encoding = None

        # add pixel_values encoding. If we also have text_encoding, update image encoding and return it.
        # else, return the text encoding.
        output_kwargs = self._merge_kwargs(
            BlipProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if text is not None:
            text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if images is not None:
            # encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
            encoding_image_processor = BatchFeature(data={"pixel_values": images}, tensor_type=output_kwargs["images_kwargs"].get("return_tensors", None))

            if text_encoding is not None:
                encoding_image_processor.update(text_encoding)
            return encoding_image_processor

        return text_encoding