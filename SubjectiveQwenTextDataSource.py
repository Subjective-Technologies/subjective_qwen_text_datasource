"""
Subjective Qwen Local Text Data Source
======================================

Text-only local Qwen LLM. A sibling of SubjectiveQwenLocalDataSource, but
without the vision encoder — for pipeline steps whose input is already
structured text (e.g. synthesizing an episode JSON from 5 atom JSONs).

Why a separate class?
- Qwen2-VL-2B is ~2.2 B params with ~700 M of those locked in the vision
  encoder. On a text-only task you pay for vision weights that contribute
  nothing, and the language head is effectively smaller than a 1.5 B
  text-only model.
- Qwen2.5-1.5B-Instruct (text-only) uses ~half the VRAM and is stronger on
  structured-input → structured-output synthesis.

Contract: identical to SubjectiveQwenLocalDataSource on the request/response
side — the node accepts a `message` and emits a `response` string. This
makes it a drop-in replacement on text-only nodes by changing only the
`class` field in the pipeline JSON.
"""

import os
from typing import Any, Dict

from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveQwenTextDataSource(SubjectiveDataSource):
    """V2 datasource for local Qwen text-only LLM inference."""

    # Per-class state (not shared with SubjectiveQwenLocalDataSource's state).
    _dependencies_available = False
    _model = None
    _tokenizer = None
    _model_loaded = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._normalize_params()
        self._check_dependencies()

    # ------------------------------------------------------------------ schemas

    @classmethod
    def connection_schema(cls):
        return {
            "model_id": {
                "type": "select",
                "label": "Model",
                "required": True,
                "default": "Qwen/Qwen2.5-1.5B-Instruct",
                "options": [
                    {"value": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B Instruct (tiniest, ~1 GB)"},
                    {"value": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B Instruct (recommended, ~3 GB)"},
                    {"value": "Qwen/Qwen2.5-3B-Instruct",   "label": "Qwen2.5 3B Instruct (~6 GB)"},
                    {"value": "Qwen/Qwen2.5-7B-Instruct",   "label": "Qwen2.5 7B Instruct (~14 GB, best quality)"},
                ],
                "description": "Text-only Qwen LLM to load locally.",
            },
            "device": {
                "type": "select",
                "label": "Device",
                "required": False,
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto (recommended)"},
                    {"value": "cuda", "label": "CUDA (GPU)"},
                    {"value": "cpu",  "label": "CPU (slow)"},
                ],
            },
            "max_new_tokens": {
                "type": "number",
                "label": "Max New Tokens",
                "required": False,
                "default": 1024,
            },
            "temperature": {
                "type": "number",
                "label": "Temperature",
                "required": False,
                "default": 0.3,
                "description": "Lower than VL default — synthesis tasks want stable output.",
            },
            "top_p": {
                "type": "number",
                "label": "Top P",
                "required": False,
                "default": 0.9,
            },
            "do_sample": {
                "type": "checkbox",
                "label": "Use Sampling",
                "required": False,
                "default": False,
                "description": "Disabled by default — greedy decoding gives stable JSON for synthesis.",
            },
            "system_prompt": {
                "type": "textarea",
                "label": "System Prompt",
                "required": False,
                "default": "",
            },
            "load_in_4bit": {
                "type": "checkbox",
                "label": "Load in 4-bit",
                "required": False,
                "default": False,
            },
            "load_in_8bit": {
                "type": "checkbox",
                "label": "Load in 8-bit",
                "required": False,
                "default": False,
            },
            "torch_dtype": {
                "type": "select",
                "label": "Torch dtype",
                "required": False,
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "bfloat16", "label": "bfloat16"},
                    {"value": "float16", "label": "float16"},
                    {"value": "float32", "label": "float32"},
                ],
            },
            "trust_remote_code": {
                "type": "checkbox",
                "label": "Trust remote code",
                "required": False,
                "default": True,
            },
        }

    @classmethod
    def request_schema(cls):
        return {
            "message": {
                "type": "textarea",
                "label": "Message",
                "required": True,
                "description": "Text prompt for the model. Wired from upstream pipeline nodes.",
            },
        }

    @classmethod
    def output_schema(cls):
        return {
            "success":  {"type": "checkbox", "label": "Success"},
            "response": {"type": "text",     "label": "Response Text"},
            "model":    {"type": "text",     "label": "Model Used"},
        }

    @classmethod
    def icon(cls) -> str:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            with open(icon_path, "r", encoding="utf-8") as fh:
                return fh.read()
        return ""

    # ----------------------------------------------------------------- chat API

    def supports_chat(self) -> bool:
        return True

    def handle_message(self, message, files=None):
        # `files` is accepted for interface symmetry with the VL data source
        # but ignored — this DS is text-only by definition.
        if isinstance(message, dict):
            payload = message.get("content", message.get("text", message.get("message", "")))
            return self._handle_request(payload)
        return self._handle_request(message)

    def run(self, request):
        if isinstance(request, dict):
            return self.handle_message(request)
        return self.handle_message(request)

    # ------------------------------------------------------------ param plumbing

    def _normalize_params(self):
        conn = getattr(self, "_connection", {}) or {}
        if not isinstance(self.params, dict):
            self.params = {}
        self.params["model_id"] = conn.get("model_id") or self.params.get("model_id") or "Qwen/Qwen2.5-1.5B-Instruct"
        self.params["device"]   = conn.get("device")   or self.params.get("device")   or "auto"

        def _num(key, default, cast, lo=None, hi=None):
            try:
                value = cast(conn.get(key) if conn.get(key) is not None else self.params.get(key, default))
            except (ValueError, TypeError):
                value = default
            if lo is not None and value < lo: value = default
            if hi is not None and value > hi: value = default
            return value

        self.params["max_new_tokens"] = _num("max_new_tokens", 1024, int, lo=1)
        self.params["temperature"]    = _num("temperature",   0.3,  float, lo=0.0, hi=2.0)
        self.params["top_p"]          = _num("top_p",         0.9,  float, lo=0.0, hi=1.0)

        for flag, default in (("do_sample", False), ("load_in_4bit", False), ("load_in_8bit", False), ("trust_remote_code", True)):
            self.params[flag] = conn[flag] if flag in conn else self.params.get(flag, default)

        self.params["torch_dtype"]   = conn.get("torch_dtype")   or self.params.get("torch_dtype", "auto")
        self.params["system_prompt"] = conn.get("system_prompt") if "system_prompt" in conn else self.params.get("system_prompt", "")

    def _check_dependencies(self):
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
            self._dependencies_available = True
            BBLogger.log("Qwen text-only dependencies available")
        except ImportError as exc:
            self._dependencies_available = False
            BBLogger.log(f"Qwen text-only dependencies missing: {exc}. Install: pip install 'transformers>=4.45' torch accelerate")

    # ---------------------------------------------------------- model lifecycle

    def _load_model(self) -> bool:
        if self._model_loaded:
            return True
        if not self._dependencies_available:
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            model_id = self.params["model_id"]
            device   = self.params["device"]

            BBLogger.log(f"Loading text-only Qwen model: {model_id}")

            dtype_str = self.params.get("torch_dtype", "auto")
            torch_dtype = {
                "bfloat16": torch.bfloat16,
                "float16":  torch.float16,
                "float32":  torch.float32,
            }.get(dtype_str, "auto")

            model_kwargs = {"trust_remote_code": self.params.get("trust_remote_code", True)}

            if self.params.get("load_in_4bit"):
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.params.get("load_in_8bit"):
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            if device == "auto":
                model_kwargs["device_map"] = "auto"
            elif device == "cuda":
                model_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                model_kwargs["device_map"] = "cpu"

            self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get("trust_remote_code", True))
            self._model_loaded = True
            BBLogger.log(f"Qwen text model loaded on device: {self._model.device}")
            return True
        except Exception as exc:
            BBLogger.log(f"Error loading Qwen text model: {exc}")
            return False

    def unload_model(self):
        try:
            import torch
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------------- inference

    def _handle_request(self, message: Any) -> Dict:
        if not self._dependencies_available:
            return {
                "success": False,
                "error": True,
                "error_type": "dependency_error",
                "message": "Required dependencies not installed. Run: pip install 'transformers>=4.45' torch accelerate",
                "original_message": message,
                "response": "",
                "model": self.params.get("model_id", ""),
            }

        if not self._load_model():
            return {
                "success": False,
                "error": True,
                "error_type": "model_load_error",
                "message": "Failed to load Qwen text model. Check logs.",
                "original_message": message,
                "response": "",
                "model": self.params.get("model_id", ""),
            }

        text = message if isinstance(message, str) else str(message or "")
        try:
            import torch

            messages = []
            system_prompt = self.params.get("system_prompt") or ""
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})

            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)

            gen_kwargs = {
                "max_new_tokens": self.params["max_new_tokens"],
                "do_sample": bool(self.params.get("do_sample", False)),
            }
            if gen_kwargs["do_sample"]:
                gen_kwargs["temperature"] = self.params["temperature"]
                gen_kwargs["top_p"] = self.params["top_p"]

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            # Strip the prompt tokens from the generated output.
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            return {
                "success": True,
                "response": response_text,
                "model": self.params["model_id"],
                "original_message": text,
            }
        except Exception as exc:
            BBLogger.log(f"Qwen text inference error: {exc}")
            return {
                "success": False,
                "error": True,
                "error_type": "inference_error",
                "message": str(exc),
                "original_message": text,
                "response": "",
                "model": self.params.get("model_id", ""),
            }
