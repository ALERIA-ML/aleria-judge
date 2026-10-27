# NVIDIA's TensorRT-LLM release image for the validated version. It ships torch,
# FlashInfer, the NVFP4 GEMM kernels for Blackwell, MPI, nvcc and the HTTP stack;
# the service adds its own code and the vision encoder's TensorRT / AITune packages.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29
# TensorRT-LLM otherwise reports usage to NVIDIA from a background thread every 10 minutes.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 TRTLLM_NO_USAGE_STATS=1
WORKDIR /app
# The vision encoder's AITune engine (JUDGE_VISION_ENGINE=aitune, the default): TensorRT 11.0, the version the image's
# torch_tensorrt 2.14.0a0 was built against (the release image ships torch_tensorrt without TensorRT's libraries), and
# NVIDIA AITune 0.6.0 with what aitune.torch imports (onnxruntime's CPU build and Model Optimizer's ONNX extras: its
# ONNX Runtime and TensorRT-via-ONNX backends are not used). The image's torch, numpy, onnx and protobuf stay as they are.
RUN pip freeze | grep -i -E "^(torch|torchvision|torchao|torch.tensorrt|nvidia-modelopt|onnx|numpy|protobuf|packaging|sympy|ml.dtypes)==" > /tmp/pins.txt \
 && pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -c /tmp/pins.txt tensorrt-cu13==11.0.0.114 \
    onnxruntime==1.31.0 onnxslim==0.1.98 onnxconverter-common==1.16.0 lief==1.0.0 cppimport==26.4.17 \
 && pip install --no-cache-dir --no-deps --extra-index-url https://pypi.nvidia.com aitune==0.6.0 autowrapt==2.0.0 \
    wrapt==2.5.1 onnx-graphsurgeon==0.6.1 \
 && rm /tmp/pins.txt
COPY aleria_judge /app/aleria_judge
# Fail the build unless TRT-LLM matches the sources the patches were validated against.
RUN python -c "from aleria_judge.patches import verify_runtime; verify_runtime()"
CMD ["python", "-m", "aleria_judge"]
