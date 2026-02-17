"""
Custom LiteLLM callback for conditional continue_final_message handling.

For Mistral models on vLLM, continue_final_message must be set to True
only when the last message is from the assistant (to continue generation).
When the last message is from the user, it must NOT be set.

This callback ONLY applies to Mistral models. Qwen models handle
consecutive assistant messages natively and don't need this.

Usage:
    litellm --config vllm_litellm_config.yaml --port 8080
"""

import sys
print("[custom_callbacks] Module loading...", file=sys.stderr)

from litellm.integrations.custom_logger import CustomLogger


class ContinueFinalMessageHandler(CustomLogger):
    """Conditionally set continue_final_message for Mistral models on vLLM."""

    # Models that need continue_final_message handling
    MISTRAL_MODELS = ("mistral", "ministral", "codestral", "pixtral")

    def __init__(self):
        super().__init__()
        print("[custom_callbacks] ContinueFinalMessageHandler initialized", file=sys.stderr)

    def _is_mistral_model(self, model: str) -> bool:
        """Check if the model is a Mistral family model."""
        model_lower = model.lower()
        return any(name in model_lower for name in self.MISTRAL_MODELS)

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type
    ):
        """
        Called just before a litellm completion call is made.
        For Mistral models, adds continue_final_message when last message is from assistant.
        """
        model = data.get("model", "")
        messages = data.get("messages", [])
        last_role = messages[-1].get("role") if messages else None

        # Only apply to Mistral models
        if not self._is_mistral_model(model):
            return data

        print(f"[custom_callbacks] Mistral model detected: {model}, last_role={last_role}", file=sys.stderr)

        if messages and last_role == "assistant":
            # Last message is from assistant - need continue_final_message
            extra_body = data.get("extra_body", {})
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
            data["extra_body"] = extra_body
            print("[custom_callbacks] Set continue_final_message=True", file=sys.stderr)

        return data


# Instance to be referenced in litellm config
proxy_handler_instance = ContinueFinalMessageHandler()
print("[custom_callbacks] proxy_handler_instance created", file=sys.stderr)
