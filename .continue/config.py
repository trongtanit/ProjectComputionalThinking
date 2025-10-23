from continuedev.libs.llm.ollama import Ollama

config = {
    "models": [
        Ollama(
            model="qwen3-coder:30b",
            api_base="http://localhost:11434"
        )
    ]
}