"""
Analysis Storage Wrapper.

Provides decorators for automatic storage and retrieval of analysis results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from maverick_mcp.data.analysis_storage import get_analysis_storage

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_analysis_storage(tool_name: str) -> Callable[[F], F]:
    """
    Decorator that adds automatic storage and retrieval to analysis functions.

    This decorator:
    1. Checks for previous analysis of the same symbol/type
    2. Executes the analysis function
    3. Stores the result in the database
    4. Enhances the result with comparison to previous analysis

    Args:
        tool_name: Name of the tool (e.g., 'technical_analysis', 'quant_analysis')

    Returns:
        Decorated function with automatic storage/retrieval

    Example:
        @with_analysis_storage("technical_analysis")
        async def technical_analysis(symbol: str, analysis_type: str = "full"):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            # Extract symbol and analysis_type from args/kwargs
            symbol = kwargs.get("symbol") or (args[0] if args else None)
            analysis_type = kwargs.get("analysis_type", "full")

            if not symbol:
                # Can't store without a symbol, just execute
                return await func(*args, **kwargs)

            storage = get_analysis_storage()

            # 1. Check for previous analysis
            previous = await storage.get_latest_analysis(
                symbol=symbol,
                tool_name=tool_name,
                analysis_type=analysis_type,
            )

            # 2. Execute the analysis function
            result = await func(*args, **kwargs)

            # 3. Store the result (only if successful)
            if result.get("status") != "error":
                # Extract input params for storage
                input_params = {k: v for k, v in kwargs.items() if k != "symbol"}
                analysis_id = await storage.store_analysis(
                    symbol=symbol,
                    tool_name=tool_name,
                    analysis_type=analysis_type,
                    result=result,
                    input_params=input_params,
                )
                if analysis_id:
                    result["analysis_id"] = analysis_id

            # 4. Enhance result with previous analysis comparison
            if previous and result.get("status") != "error":
                comparison = storage.compare_analyses(
                    previous=previous,
                    current=result,
                    tool_name=tool_name,
                    analysis_type=analysis_type,
                )
                result["previous_analysis"] = comparison

            return result

        return wrapper  # type: ignore

    return decorator


def with_analysis_history(tool_name: str, max_history: int = 5) -> Callable[[F], F]:
    """
    Decorator that includes recent analysis history in results.

    This is a lighter-weight alternative to with_analysis_storage
    that only retrieves history, without storing the current result.

    Args:
        tool_name: Name of the tool
        max_history: Maximum number of historical analyses to include

    Returns:
        Decorated function with history included
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            symbol = kwargs.get("symbol") or (args[0] if args else None)

            # Execute the function
            result = await func(*args, **kwargs)

            # Add history if we have a symbol
            if symbol and result.get("status") != "error":
                storage = get_analysis_storage()
                history = await storage.get_analysis_history(
                    symbol=symbol,
                    tool_name=tool_name,
                    limit=max_history,
                )
                if history:
                    result["analysis_history"] = {
                        "count": len(history),
                        "recent": [
                            {
                                "date": h["created_at"],
                                "type": h["analysis_type"],
                                "summary": _extract_summary(h.get("result", {})),
                            }
                            for h in history
                        ],
                    }

            return result

        return wrapper  # type: ignore

    return decorator


def _extract_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extract key summary fields from an analysis result."""
    summary_keys = [
        "signal",
        "trend",
        "outlook",
        "rating",
        "score",
        "composite_score",
        "recommendation",
        "current_price",
        "beta",
        "current_rsi",
    ]
    return {k: v for k, v in result.items() if k in summary_keys and v is not None}
