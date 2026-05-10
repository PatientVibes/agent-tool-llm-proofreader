"""LLM-powered OCR proofreading agent (langchain-openai via OpenRouter; supports an alternate LangGraph-based runner).

State files live at platformdirs.user_state_dir('agent-tool-llm-proofreader') by default;
override with the AGENT_TOOL_PROOFREADER_STATE_DIR env var or --state-dir CLI flag.
"""
__version__ = "0.1.0"
