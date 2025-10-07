import logging
import os
import opentelemetry.context as context_api
from importlib import import_module, metadata
from functools import wraps
from packaging.version import Version

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import Tracer, TracerProvider as iTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from snowglobe.client.src.models import (
    CompletionRequest,
    CompletionFunctionOutputs,
    RiskEvaluationRequest,
    RiskEvaluationOutputs,
)
from snowglobe.client.src.types import (
    CompletionFnTelemetryContext,
    RiskEvalTelemetryContext,
)
from snowglobe.telemetry.openinference.openinference_utils import _flatten

from typing import Any, Awaitable, Callable, Collection, TypeVar, Union


logger = logging.getLogger(__name__)


T = TypeVar("T")


class OpenInferenceInstrumentor(BaseInstrumentor):
    __slots__ = (
        "_tracer",
        "_snowglobe_version",
        "_run_completion_fn",
        "_run_risk_evaluation_fn",
    )

    _trace: Tracer
    _snowglobe_version: str
    _run_completion_fn: Callable[
        [
            Union[
                Callable[[CompletionRequest], CompletionFunctionOutputs],
                Callable[[CompletionRequest], Awaitable[CompletionFunctionOutputs]],
            ],
            CompletionRequest,
            CompletionFnTelemetryContext,
        ],
        Awaitable[CompletionFunctionOutputs],
    ]
    _run_risk_evaluation_fn: Callable[
        [
            Union[
                Callable[[RiskEvaluationRequest], RiskEvaluationOutputs],
                Callable[[RiskEvaluationRequest], Awaitable[RiskEvaluationOutputs]],
            ],
            RiskEvaluationRequest,
            RiskEvalTelemetryContext,
        ],
        Awaitable[RiskEvaluationOutputs],
    ]

    def __init__(self):
        super().__init__()
        self._snowglobe_version = metadata.version("snowglobe")
        runner = import_module("snowglobe.client.src.runner")
        self._run_completion_fn = runner.run_completion_fn
        self._run_risk_evaluation_fn = runner.run_risk_evaluation_fn

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["snowglobe >= 0.4.16"]

    def _setup_default_tracer_provider(self) -> iTracerProvider:
        tracer_provider = trace_api.get_tracer_provider()

        if (
            not tracer_provider._active_span_processor._span_processors  # type: ignore
            and hasattr(tracer_provider, "add_span_processor")
        ):
            if not os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL"):
                os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
            trace_exporter = OTLPSpanExporter()
            span_processor = BatchSpanProcessor(trace_exporter)
            tracer_provider.add_span_processor(span_processor)  # type: ignore

        return tracer_provider

    def _instrument(self, **kwargs: Any):
        version = Version(metadata.version("snowglobe"))
        if (version.major, version.minor, version.micro) < (0, 4, 16):
            logger.info("Snowglobe version < 0.4.16 detected, skipping instrumentation")
            return

        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            tracer_provider = self._setup_default_tracer_provider()

        config = kwargs.get("config")
        if not config:
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, self._snowglobe_version, tracer_provider),
            config=config,
        )

        runner = import_module("snowglobe.client.src.runner")

        run_completion_fn: Callable[
            [
                Union[
                    Callable[[CompletionRequest], CompletionFunctionOutputs],
                    Callable[[CompletionRequest], Awaitable[CompletionFunctionOutputs]],
                ],
                CompletionRequest,
                CompletionFnTelemetryContext,
            ],
            Awaitable[CompletionFunctionOutputs],
        ] = runner.run_completion_fn
        wrapped_run_completion_fn = self._instrument_completion_fn(run_completion_fn)
        setattr(runner, "run_completion_fn", wrapped_run_completion_fn)
        setattr(runner.run_completion_fn, "__instrumented_by_openinference", True)

        run_risk_evaluation_fn: Callable[
            [
                Union[
                    Callable[[RiskEvaluationRequest], RiskEvaluationOutputs],
                    Callable[[RiskEvaluationRequest], Awaitable[RiskEvaluationOutputs]],
                ],
                RiskEvaluationRequest,
                RiskEvalTelemetryContext,
            ],
            Awaitable[RiskEvaluationOutputs],
        ] = runner.run_risk_evaluation_fn
        wrapped_risk_evaluation_fn = self._instrument_risk_evaluation_fn(
            run_risk_evaluation_fn
        )
        setattr(runner, "run_risk_evaluation_fn", wrapped_risk_evaluation_fn)
        setattr(runner.run_risk_evaluation_fn, "__instrumented_by_openinference", True)

    def _uninstrument(self, **kwargs: Any):
        runner = import_module("snowglobe.client.src.runner")
        if self._run_completion_fn:
            setattr(runner, "run_completion_fn", self._run_completion_fn)
            delattr(runner.run_completion_fn, "__instrumented_by_openinference")

        if self._run_risk_evaluation_fn:
            setattr(runner, "run_risk_evaluation_fn", self._run_risk_evaluation_fn)
            delattr(runner.run_risk_evaluation_fn, "__instrumented_by_openinference")

    async def _collapse(self, wave_fn: Union[T, Awaitable[T]]) -> T:
        if isinstance(wave_fn, Awaitable):
            awaited = await wave_fn
            return awaited
        return wave_fn

    def _instrument_completion_fn(
        self,
        run_completion_fn: Callable[
            [
                Union[
                    Callable[[CompletionRequest], CompletionFunctionOutputs],
                    Callable[[CompletionRequest], Awaitable[CompletionFunctionOutputs]],
                ],
                CompletionRequest,
                CompletionFnTelemetryContext,
            ],
            Awaitable[CompletionFunctionOutputs],
        ],
    ):
        @wraps(run_completion_fn)
        async def run_completion_fn_wrapper(
            completion_fn: Union[
                Callable[[CompletionRequest], CompletionFunctionOutputs],
                Callable[[CompletionRequest], Awaitable[CompletionFunctionOutputs]],
            ],
            completion_request: CompletionRequest,
            telemetry_context: CompletionFnTelemetryContext,
        ) -> CompletionFunctionOutputs:
            session_id = telemetry_context["session_id"]
            conversation_id = telemetry_context["conversation_id"]
            message_id = telemetry_context["message_id"]
            simulation_name = telemetry_context["simulation_name"]
            agent_name = telemetry_context["agent_name"]
            span_type = telemetry_context["span_type"]

            @wraps(completion_fn)
            async def completion_fn_wrapper(
                req: CompletionRequest,
            ) -> CompletionFunctionOutputs:
                if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                    completion_fn_out = await self._collapse(completion_fn(req))
                else:
                    messages = req.to_openai_messages()
                    with self._tracer.start_as_current_span(
                        span_type,
                        attributes=dict(
                            _flatten(
                                {
                                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM,
                                    SpanAttributes.INPUT_VALUE: messages[-1].get(
                                        "content"
                                    )
                                    or "",
                                    SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                                }
                            )
                        ),
                    ) as span:
                        context_attributes = dict(get_attributes_from_context())
                        span_attributes = dict(
                            _flatten(
                                {
                                    **context_attributes,
                                    SpanAttributes.SESSION_ID: str(session_id),
                                    SpanAttributes.PROMPT_ID: str(message_id),
                                    SpanAttributes.AGENT_NAME: agent_name,
                                    SpanAttributes.LLM_INPUT_MESSAGES: messages,
                                    "snowglobe.version": self._snowglobe_version,
                                    "snowglobe.span.type": span_type,
                                    "snowglobe.conversation.id": str(conversation_id),
                                    "snowglobe.message.id": str(message_id),
                                    "snowglobe.simulation.name": simulation_name,
                                }
                            )
                        )
                        span.set_attributes(span_attributes)
                        try:
                            completion_fn_out = await self._collapse(completion_fn(req))
                            span.set_attributes(
                                dict(
                                    _flatten(
                                        {
                                            SpanAttributes.OUTPUT_VALUE: completion_fn_out.response,
                                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                                            SpanAttributes.LLM_OUTPUT_MESSAGES: [
                                                *messages,
                                                {
                                                    "role": "assistant",
                                                    "content": completion_fn_out.response,
                                                },
                                            ],
                                        }
                                    )
                                )
                            )
                        except Exception as exception:
                            span.set_status(
                                trace_api.Status(
                                    trace_api.StatusCode.ERROR, str(exception)
                                )
                            )
                            span.record_exception(exception)
                            raise
                        span.set_status(trace_api.StatusCode.OK)

                return completion_fn_out

            completion_fn_wrapper_out = await completion_fn_wrapper(completion_request)
            return completion_fn_wrapper_out

        return run_completion_fn_wrapper

    def _instrument_risk_evaluation_fn(
        self,
        run_risk_evaluation_fn: Callable[
            [
                Union[
                    Callable[[RiskEvaluationRequest], RiskEvaluationOutputs],
                    Callable[[RiskEvaluationRequest], Awaitable[RiskEvaluationOutputs]],
                ],
                RiskEvaluationRequest,
                RiskEvalTelemetryContext,
            ],
            Awaitable[RiskEvaluationOutputs],
        ],
    ):
        @wraps(run_risk_evaluation_fn)
        async def run_risk_evaluation_fn_wrapper(
            risk_evaluation_fn: Union[
                Callable[[RiskEvaluationRequest], RiskEvaluationOutputs],
                Callable[[RiskEvaluationRequest], Awaitable[RiskEvaluationOutputs]],
            ],
            risk_evaluation_request: RiskEvaluationRequest,
            telemetry_context: RiskEvalTelemetryContext,
        ) -> RiskEvaluationOutputs:
            session_id = telemetry_context["session_id"]
            conversation_id = telemetry_context["conversation_id"]
            message_id = telemetry_context["message_id"]
            simulation_name = telemetry_context["simulation_name"]
            agent_name = telemetry_context["agent_name"]
            span_type = telemetry_context["span_type"]
            risk_name = telemetry_context["risk_name"]

            @wraps(risk_evaluation_fn)
            async def risk_evaluation_fn_wrapper(
                req: RiskEvaluationRequest,
            ):
                if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                    risk_evaluation_fn_out = await self._collapse(
                        risk_evaluation_fn(req)
                    )
                else:
                    with self._tracer.start_as_current_span(
                        span_type,
                        attributes=dict(
                            _flatten(
                                {
                                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EVALUATOR,
                                    SpanAttributes.INPUT_VALUE: req.model_dump_json(),
                                    SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                                }
                            )
                        ),
                    ) as span:
                        context_attributes = dict(get_attributes_from_context())
                        span_attributes = {
                            **context_attributes,
                            SpanAttributes.SESSION_ID: str(session_id),
                            SpanAttributes.PROMPT_ID: str(message_id),
                            SpanAttributes.AGENT_NAME: agent_name,
                            "snowglobe.client.version": self._snowglobe_version,
                            "snowglobe.span.type": span_type,
                            "snowglobe.conversation.id": str(conversation_id),
                            "snowglobe.message.id": str(message_id),
                            "snowglobe.simulation.name": simulation_name,
                            "snowglobe.risk.name": risk_name,
                        }
                        span.set_attributes(span_attributes)
                        try:
                            risk_evaluation_fn_out = await self._collapse(
                                risk_evaluation_fn(req)
                            )
                            span.set_attributes(
                                dict(
                                    _flatten(
                                        {
                                            SpanAttributes.OUTPUT_VALUE: risk_evaluation_fn_out.model_dump_json(),
                                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                                        }
                                    )
                                )
                            )
                        except Exception as exception:
                            span.set_status(
                                trace_api.Status(
                                    trace_api.StatusCode.ERROR, str(exception)
                                )
                            )
                            span.record_exception(exception)
                            raise
                        span.set_status(trace_api.StatusCode.OK)

                return risk_evaluation_fn_out

            risk_evaluation_fn_wrapper_out = await risk_evaluation_fn_wrapper(
                risk_evaluation_request
            )
            return risk_evaluation_fn_wrapper_out

        return run_risk_evaluation_fn_wrapper
