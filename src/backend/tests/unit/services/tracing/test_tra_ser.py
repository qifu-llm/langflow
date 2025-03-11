import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langflow.services.settings.base import Settings
from langflow.services.settings.service import SettingsService
from langflow.services.tracing.base import BaseTracer
from langflow.services.tracing.service import TracingService


class MockTracer(BaseTracer):
    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: uuid.UUID,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.project_name = project_name
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self._ready = True
        self.add_trace_called = False
        self.end_trace_called = False
        self.end_called = False
        self.get_langchain_callback_called = False
        self.add_trace_list = []
        self.end_trace_list = []

    @property
    def ready(self) -> bool:
        return self._ready

    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, any],
        metadata: dict[str, any] | None = None,
        vertex=None,
    ) -> None:
        self.add_trace_list.append({
            "trace_id": trace_id,
            "trace_name": trace_name,
            "trace_type": trace_type,
            "inputs": inputs,
            "metadata": metadata,
            "vertex": vertex
        })

    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, any] | None = None,
        error: Exception | None = None,
        logs=(),
    ) -> None:
        self.end_trace_list.append({
            "trace_id": trace_id,
            "trace_name": trace_name,
            "outputs": outputs,
            "error": error,
            "logs": logs
        })

    def end(
        self,
        inputs: dict[str, any],
        outputs: dict[str, any],
        error: Exception | None = None,
        metadata: dict[str, any] | None = None,
    ) -> None:
        self.end_called = True
        self.inputs_param = inputs
        self.outputs_param = outputs
        self.error_param = error
        self.metadata_param = metadata

    def get_langchain_callback(self):
        self.get_langchain_callback_called = True
        return MagicMock()


@pytest.fixture
def mock_settings_service():
    settings = Settings()
    settings.deactivate_tracing = False
    settings_service = SettingsService(settings, MagicMock())
    return settings_service


@pytest.fixture
def tracing_service(mock_settings_service: SettingsService) -> TracingService:
    service = TracingService(mock_settings_service)
    return service


@pytest.fixture
def mock_component():
    component = MagicMock()
    component._vertex = MagicMock()
    component._vertex.id = "test_vertex_id"
    component.trace_type = "test_trace_type"
    return component


@pytest.mark.asyncio
async def test_concurrent_tracing(tracing_service: TracingService, mock_component: MagicMock):
    """测试两个Task同时进行start_tracers，每个Task内部再启动2个task同时trace_component"""
    
    with patch("langflow.services.tracing.service._get_langsmith_tracer", return_value=MockTracer), \
         patch("langflow.services.tracing.service._get_langwatch_tracer", return_value=MockTracer), \
         patch("langflow.services.tracing.service._get_langfuse_tracer", return_value=MockTracer), \
         patch("langflow.services.tracing.service._get_arize_phoenix_tracer", return_value=MockTracer):
        
        # 定义通用任务函数：启动跟踪器并运行两个组件跟踪
        async def run_task(run_id, run_name, user_id, session_id, project_name, inputs, metadata, task_prefix, sleep_duration=0.1):
            tracing_service.run_id = run_id
            tracing_service.run_name = run_name
            tracing_service.user_id = user_id
            tracing_service.session_id = session_id
            tracing_service.project_name = project_name
            await tracing_service.initialize_tracers()
            async def run_component_task(component, trace_name, component_suffix):
                    async with tracing_service.trace_context(component, trace_name, inputs, metadata) as ts:
                        ts.add_log(trace_name, {"message": f"{task_prefix} {component_suffix} log"})
                        outputs = {"output_key": f"{task_prefix}_{component_suffix}_output"}
                        await asyncio.sleep(sleep_duration)
                        ts.set_outputs(trace_name, outputs)

            task1 = asyncio.create_task(run_component_task(mock_component, f"{run_id} trace_name1", f"{run_id} component1"))
            await task1
            task2 = asyncio.create_task(run_component_task(mock_component, f"{run_id} trace_name2", f"{run_id} component2"))
            await task2
            
            await tracing_service.end({"final_output": f"{task_prefix}_final_output"})
            print(f"{run_id} end")
            return tracing_service._tracers['langfuse']
        
        inputs1 = {"input_key": "input_value1"}
        metadata1 = {"metadata_key": "metadata_value1"}
        inputs2 = {"input_key": "input_value2"}
        metadata2 = {"metadata_key": "metadata_value2"}
        
        task1 = asyncio.create_task(run_task("run_id1", "run_name1", "user_id1", "session_id1", "project_name1", inputs1, metadata1, "task1", 2))
        await asyncio.sleep(0.1)
        task2 = asyncio.create_task(run_task("run_id2", "run_name2", "user_id2", "session_id2", "project_name2", inputs2, metadata2, "task2", 0.1))

        tracer1 = await task1
        tracer2 = await task2
        
        # Verify tracer1 and tracer2 have correct trace data
        print(f"tracer1 == tracer2: {tracer1 == tracer2}")
        print(f"tracer1: {len(tracer1.end_trace_list)}")
        print(f"tracer1: {tracer1.end_trace_list}")
        print(f"tracer2: {tracer2.end_trace_list}")

        assert tracer1 != tracer2

