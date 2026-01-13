"""Unified Watchlist Tool.

Single tool for all watchlist management operations:
- create: Create a new watchlist
- list: List all watchlists
- view: View watchlist with current prices
- add: Add ticker(s) to watchlist
- remove: Remove ticker from watchlist
- delete: Delete entire watchlist
- performance: Get performance summary
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def watchlist_manage(
    action: str,
    name: str | None = None,
    ticker: str | None = None,
    tickers: list[str] | None = None,
    target_price: float | None = None,
    stop_loss: float | None = None,
    notes: str | None = None,
    description: str | None = None,
    user_id: str = "default",
    confirm: bool = False,
) -> dict[str, Any]:
    """
    Unified watchlist management for creating, viewing, and managing watchlists.

    Consolidates all watchlist operations into a single tool with an action parameter.

    Args:
        action: Watchlist action to perform:
            - 'create': Create a new watchlist (requires name)
            - 'list': List all watchlists for user
            - 'view': View watchlist items with current prices (requires name)
            - 'add': Add ticker(s) to watchlist (requires name + ticker or tickers)
            - 'remove': Remove ticker from watchlist (requires name + ticker)
            - 'delete': Delete entire watchlist (requires name + confirm=True)
            - 'performance': Get performance summary (requires name)
        name: Watchlist name (required for most actions)
        ticker: Single stock ticker (for add/remove)
        tickers: List of stock tickers (for bulk add)
        target_price: Target price for alerts (optional, for add)
        stop_loss: Stop loss price (optional, for add)
        notes: Notes about the ticker (optional, for add)
        description: Watchlist description (optional, for create)
        user_id: User identifier (default: "default")
        confirm: Safety confirmation for delete action

    Returns:
        Dictionary containing action result.

    Examples:
        # Create a watchlist
        >>> watchlist_manage("create", name="Tech Growth")

        # List all watchlists
        >>> watchlist_manage("list")

        # Add a stock with target price
        >>> watchlist_manage("add", name="Tech Growth", ticker="NVDA", target_price=200)

        # Add multiple stocks
        >>> watchlist_manage("add", name="Tech Growth", tickers=["AAPL", "MSFT", "GOOGL"])

        # View watchlist with prices
        >>> watchlist_manage("view", name="Tech Growth")

        # Get performance summary
        >>> watchlist_manage("performance", name="Tech Growth")

        # Remove a stock
        >>> watchlist_manage("remove", name="Tech Growth", ticker="AAPL")

        # Delete watchlist
        >>> watchlist_manage("delete", name="Tech Growth", confirm=True)
    """
    action = action.lower().strip()

    valid_actions = ["create", "list", "view", "add", "remove", "delete", "performance"]
    if action not in valid_actions:
        return {
            "error": f"Invalid action '{action}'. Must be one of: {valid_actions}",
            "status": "error",
        }

    try:
        from maverick_mcp.domain.watchlist import (
            add_multiple_to_watchlist,
            add_to_watchlist,
            create_watchlist,
            delete_watchlist,
            get_watchlist,
            get_watchlist_performance,
            list_watchlists,
            remove_from_watchlist,
        )

        if action == "create":
            if not name:
                return {
                    "error": "name is required for create action",
                    "hint": "Example: watchlist_manage('create', name='Tech Growth')",
                    "status": "error",
                }
            result = create_watchlist(
                name=name,
                user_id=user_id,
                description=description,
            )
            result["action"] = "create"
            return result

        elif action == "list":
            result = list_watchlists(user_id=user_id)
            result["action"] = "list"
            return result

        elif action == "view":
            if not name:
                return {
                    "error": "name is required for view action",
                    "hint": "Example: watchlist_manage('view', name='Tech Growth')",
                    "status": "error",
                }
            result = get_watchlist(
                name=name,
                user_id=user_id,
                include_prices=True,
            )
            result["action"] = "view"
            return result

        elif action == "add":
            if not name:
                return {
                    "error": "name is required for add action",
                    "hint": "Example: watchlist_manage('add', name='Tech Growth', ticker='AAPL')",
                    "status": "error",
                }
            if not ticker and not tickers:
                return {
                    "error": "ticker or tickers is required for add action",
                    "hint": "Example: watchlist_manage('add', name='Tech Growth', ticker='AAPL')",
                    "status": "error",
                }

            # Single ticker add
            if ticker and not tickers:
                result = add_to_watchlist(
                    name=name,
                    ticker=ticker,
                    user_id=user_id,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    notes=notes,
                )
                result["action"] = "add"
                return result

            # Multiple tickers add
            if tickers:
                result = add_multiple_to_watchlist(
                    name=name,
                    tickers=tickers,
                    user_id=user_id,
                )
                result["action"] = "add"
                return result

        elif action == "remove":
            if not name or not ticker:
                return {
                    "error": "name and ticker are required for remove action",
                    "hint": "Example: watchlist_manage('remove', name='Tech Growth', ticker='AAPL')",
                    "status": "error",
                }
            result = remove_from_watchlist(
                name=name,
                ticker=ticker,
                user_id=user_id,
            )
            result["action"] = "remove"
            return result

        elif action == "delete":
            if not name:
                return {
                    "error": "name is required for delete action",
                    "hint": "Example: watchlist_manage('delete', name='Tech Growth', confirm=True)",
                    "status": "error",
                }
            result = delete_watchlist(
                name=name,
                user_id=user_id,
                confirm=confirm,
            )
            result["action"] = "delete"
            return result

        elif action == "performance":
            if not name:
                return {
                    "error": "name is required for performance action",
                    "hint": "Example: watchlist_manage('performance', name='Tech Growth')",
                    "status": "error",
                }
            result = get_watchlist_performance(
                name=name,
                user_id=user_id,
            )
            result["action"] = "performance"
            return result

        else:
            return {"error": f"Unknown action: {action}", "status": "error"}

    except Exception as e:
        logger.error(f"Error in watchlist_manage: {e}")
        return {
            "error": str(e),
            "action": action,
            "status": "error",
        }
