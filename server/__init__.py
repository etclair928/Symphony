# =================================================================
# MODULE: server/__init__.py
# DESCRIPTION: WebSocket server for real-time transcription.
#
# Authored by: {DeepSeek} - Server package (2026-05-09)
# =================================================================

"""
WebSocket server for Grimlock 5.0.

Provides real-time audio transcription over WebSocket.
"""

from server.websocket import WebSocketServer, MessageType, ClientSession

__all__ = [
    "WebSocketServer",
    "MessageType",
    "ClientSession"
]