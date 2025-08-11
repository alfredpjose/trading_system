# core/events.py
import asyncio
from collections import defaultdict
from typing import Set, Dict
from loguru import logger

class EventBus(IEventBus):
    """Async event bus for system communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, Set[asyncio.Queue]] = defaultdict(set)
        self._running = False
    
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        if not self._running:
            return
            
        subscribers = self._subscribers.get(event.event_type, set())
        if subscribers:
            await asyncio.gather(
                *[queue.put(event) for queue in subscribers],
                return_exceptions=True
            )
    
    async def subscribe(self, event_type: EventType) -> AsyncGenerator[Event, None]:
        """Subscribe to events of specific type"""
        queue = asyncio.Queue(maxsize=1000)
        self._subscribers[event_type].add(queue)
        
        try:
            while self._running:
                event = await queue.get()
                yield event
        finally:
            self._subscribers[event_type].discard(queue)
    
    async def unsubscribe(self, event_type: EventType):
        """Clear all subscribers for event type"""
        self._subscribers[event_type].clear()
    
    async def start(self):
        """Start event bus"""
        self._running = True
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop event bus"""
        self._running = False
        # Clear all queues
        for event_type in self._subscribers:
            for queue in self._subscribers[event_type]:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
        logger.info("Event bus stopped")