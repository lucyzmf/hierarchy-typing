# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import pandas as pd
import pydantic
from exca.utils import DISCRIMINATOR_FIELD


logger = logging.getLogger(__name__)


class BaseEnhancer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: str
    # for automatic registration
    _CLASSES: tp.ClassVar[dict[str, type["BaseEnhancer"]]] = {}
    _discriminating_type_adapter: tp.ClassVar[pydantic.TypeAdapter]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        # check params
        super().__pydantic_init_subclass__(**kwargs)
        name = cls.__name__
        if "Base" not in name and not name.startswith("_"):
            if "name" not in cls.model_fields or cls.model_fields["name"].default != name:  # type: ignore
                # unfortunately, this field can't be added dynamically so far :(
                # https://github.com/pydantic/pydantic/issues/1937
                indication = f"name: tp.Literal[{name!r}] = {name!r}"
                raise NotImplementedError(
                    f"Enhancer {name} has incorret/missing name field, add:\n{indication}"
                )
            BaseEnhancer._CLASSES[cls.model_fields["name"].default] = cls  # type: ignore
            # from https://github.com/pydantic/pydantic/issues/7366:
            # The following will create a new type adapter every time a new subclass is created,
            # which is fine if there aren't that many classes (as far as performance goes)
            BaseEnhancer._discriminating_type_adapter = pydantic.TypeAdapter(
                tp.Annotated[
                    tp.Union[tuple(cls._CLASSES.values())],
                    pydantic.Field(discriminator="name"),
                ]
            )

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _parse_into_subclass(
        cls, v: tp.Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> "BaseEnhancer":
        if cls is BaseEnhancer:
            out = BaseEnhancer._discriminating_type_adapter.validate_python(v)
        else:
            out = handler(v)
        # make sure it is discriminated (keep name field in uid)
        out.__dict__[DISCRIMINATOR_FIELD] = "name"
        return out

    @pydantic.model_serializer
    def _dump(self, info: pydantic.SerializationInfo) -> dict[str, tp.Any]:
        # replace defaults dumper that would ignore non BaseEnhancer fields
        out: dict[str, tp.Any] = {}
        for name, field in self.model_fields.items():
            val = getattr(self, name)
            if not info.exclude_defaults or val != field.default:
                out[name] = val
        return out

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


EnhancerConfig = BaseEnhancer
Enhancer = BaseEnhancer


# # # # # tooling # # # # #


