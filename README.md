# Snowlgobe Telemetry Instrumentation for OpenInference

Instrument your Snowglobe connected app with OpenInference and start sending traces to popular OpenInference compatible sinks like Arize or Arize Phoenix.

## Installation

```
pip install snowglobe-telemetry-openinference
```

If using uv, set the `--prerelease=allow` flag
```
uv pip install --prerelease=allow snowglobe-telemetry-openinference
```


## Add the OpenInferenceInstrumentor to your agent file

Reminder: Each agent wrapper file resides in the root directory of your project, and is named after the agent (e.g. `My Agent Name` becomes `my_agent_name.py`).

```python
from snowglobe.client import CompletionRequest, CompletionFunctionOutputs
from openai import OpenAI
import os

os.env["OTEL_PYTHON_TRACER_PROVIDER"] = "sdk_tracer_provider"

### Add these two lines to your agent file and watch context rich traces come in!
from snowglobe.telemetry.openinference import OpenInferenceInstrumentor
OpenInferenceInstrumentor().instrument()


client = OpenAI(api_key=os.getenv("SNOWGLOBE_API_KEY"))

def completion_fn(request: CompletionRequest) -> CompletionFunctionOutputs:
    """
    Process a scenario request from Snowglobe.
    
    This function is called by the Snowglobe client to process requests. It should return a
    CompletionFunctionOutputs object with the response content.

    Example CompletionRequest:
    CompletionRequest(
        messages=[
            SnowglobeMessage(role="user", content="Hello, how are you?", snowglobe_data=None),
        ]
    )

    Example CompletionFunctionOutputs:
    CompletionFunctionOutputs(response="This is a string response from your application")
    
    Args:
        request (CompletionRequest): The request object containing the messages.

    Returns:
        CompletionFunctionOutputs: The response object with the generated content.
    """

    # Process the request using the messages. Example:
    messages = request.to_openai_messages()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return CompletionFunctionOutputs(response=response.choices[0].message.content)
```



## Enhancing Snowglobe Connect SDK's Traces with OpenInference Integrations
You can add more rich context to the traces the Snowglobe Connect SDK captures by installing additional OpenInference instrumentors and registering the appropriate tracer provider in your agent wrapper file.  

The below examples shows how to add OpenAI instrumentation for either Arize or Arize Phoenix in addition to Snowglobe's OpenInference instrumentation:

### Arize

Install the Arize OpenTelemetry pacakge and the OpenAI specific instrumentor.
```sh
pip install openinference-instrumentation-openai arize-otel
```

Then register the tracer provider and use the OpenAI instrumentator in your agent file:
```py
import os
from openai import OpenAI
from snowglobe.client import CompletionRequest, CompletionFunctionOutputs
from arize.otel import register

# Setup OTel via our convenience function
tracer_provider = register(
    space_id = "your-space-id", # in app space settings page
    api_key = "your-api-key", # in app space settings page
    project_name = "your-project-name", # name this to whatever you would like
)

# Import the OpenAI instrumentor from OpenInference
from openinference.instrumentation.openai import OpenAIInstrumentor

# Instrument OpenAI
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Import the OpenInference instrumentor from Snowglobe
from snowglobe.telemetry.openinference import OpenInferenceInstrumentor

# Instrument the Snowglobe client
OpenInferenceInstrumentor().instrument(tracer_provider=tracer_provider)


def completion_fn(request: CompletionRequest) -> CompletionFunctionOutputs:
    messages = request.to_openai_messages()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return CompletionFunctionOutputs(response=response.choices[0].message.content)
```


### Arize Phoenix

Install the Arize Phoenix OpenTelemetry pacakge and the OpenAI specific instrumentor.
```sh
pip install openinference-instrumentation-openai arize-phoenix-otel
```

Then register the tracer provider and use the OpenAI instrumentator in your agent file:
```py
import os
from openai import OpenAI
from snowglobe.client import CompletionRequest, CompletionFunctionOutputs
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

# configure the Phoenix tracer
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
)

# Import the OpenAI instrumentor from OpenInference
from openinference.instrumentation.openai import OpenAIInstrumentor

# Instrument OpenAI
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Import the OpenInference instrumentor from Snowglobe
from snowglobe.telemetry.openinference import OpenInferenceInstrumentor

# Instrument the Snowglobe client
OpenInferenceInstrumentor().instrument(tracer_provider=tracer_provider)


def completion_fn(request: CompletionRequest) -> CompletionFunctionOutputs:
    messages = request.to_openai_messages()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return CompletionFunctionOutputs(response=response.choices[0].message.content)
```