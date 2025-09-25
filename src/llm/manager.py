from typing import Optional, Union, List, Any, cast, Type

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from pydantic.fields import FieldInfo


class FieldDefinition(BaseModel):
    """
    Definition of a field in a signature.

    :param name: The name of the field.
    :param field_type: The type of the field (field annotation).
    :param description: Description of the field.
    """
    name: str
    field_type: Any
    description: str


def create_pydantic_model(fields: list[FieldDefinition]) -> Type[BaseModel]:
    keys = [field.name for field in fields]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate field names found")

    annotations, params = {}, {}

    for item in fields:
        params[item.name] = FieldInfo(description=item.description)
        annotations[item.name] = item.field_type

    output = type("StructuredResponse", (BaseModel,), {"__annotations__": annotations, **params})
    return cast(Type[BaseModel], output)


class SamplingParams(BaseModel):
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0

    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    seed: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None

    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None


class TaskManager:
    def __init__(
            self,
            model: str,
            guided_decoding_backend: Optional[str] = None,
            **kwargs
    ):
        self.model = model
        self.client = OpenAI(**kwargs)
        self.async_client = AsyncOpenAI(**kwargs)
        self.guided_decoding_backend = guided_decoding_backend

    async def generate_async(
            self,
            user_prompt: str,
            response_format: Optional[Union[type[BaseModel], list[FieldDefinition]]] = None,
            system_prompt: str = "You are a helpful assistant.",
            sampling_params: Optional[SamplingParams] = None
    ) -> Union[BaseModel, List[BaseModel], str, List[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        messages, response_format, extra_body = self._get_generation_parameters(
            user_prompt=user_prompt,
            response_format=response_format,
            system_prompt=system_prompt
        )

        if response_format is not None:
            completion = await self.async_client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **sampling_params.model_dump(),
                extra_body=extra_body,
            )
        else:
            completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **sampling_params.model_dump()
            )

        response = self._parse_response(
            chat_completion=completion,
            response_format=response_format,
            sampling_params=sampling_params
        )
        return response

    def generate(
            self,
            user_prompt: str,
            response_format: Optional[Union[type[BaseModel], list[FieldDefinition]]] = None,
            system_prompt: str = "You are a helpful assistant.",
            sampling_params: Optional[SamplingParams] = None
    ) -> Union[BaseModel, List[BaseModel], str, List[str]]:

        if sampling_params is None:
            sampling_params = SamplingParams()

        messages, response_format, extra_body = self._get_generation_parameters(
            user_prompt=user_prompt,
            response_format=response_format,
            system_prompt=system_prompt
        )

        if response_format is not None:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **sampling_params.model_dump(),
                extra_body=extra_body,
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **sampling_params.model_dump()
            )

        response = self._parse_response(
            chat_completion=completion,
            response_format=response_format,
            sampling_params=sampling_params
        )
        return response

    def _get_generation_parameters(
            self,
            user_prompt: str,
            response_format: Optional[Union[type[BaseModel], list[FieldDefinition]]] = None,
            system_prompt: str = "You are a helpful assistant.",
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if isinstance(response_format, list):
            for item in response_format:
                if not isinstance(item, FieldDefinition):
                    raise ValueError('Response format should be a list of `FieldDefinition` items')
            response_format = create_pydantic_model(fields=response_format)

        if self.guided_decoding_backend is not None:
            extra_body = dict(guided_decoding_backend=self.guided_decoding_backend),
        else:
            extra_body = {}

        return messages, response_format, extra_body

    def _parse_response(
            self,
            chat_completion: ChatCompletion,
            response_format: Optional[Union[type[BaseModel], list[FieldDefinition]]],
            sampling_params: SamplingParams
    ) -> Union[BaseModel, List[BaseModel], str, List[str], List[dict[str, float]]]:
        """
        Parse the response from the OpenAI API.
        """
        if sampling_params.logprobs is None:
            response = [choice.message for choice in chat_completion.choices]

            if response_format is not None:
                response = [msg.parsed for msg in response]
            else:
                response = [msg.content for msg in response]

            if sampling_params.n == 1:
                return response[0]
            else:
                return response
        else:
            response = []
            for completion_logprob_info in chat_completion.choices[0].logprobs.content:
                response.append({top_logprob.token.strip(): top_logprob.logprob
                                 for top_logprob in completion_logprob_info.top_logprobs})

            return response
