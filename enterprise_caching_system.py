"""
Enterprise-Grade Caching System for Inyandiko Legal AI Assistant
Multi-layer caching with Redis, memory, and disk storage
"""

import os
import json
import pickle
import hashlib
import logging
import asyncio
import time
import gzip
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable, AsyncGenerator, cast
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

# Redis and caching
import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError
import diskcache

# Compression and serialization
import zlib
import lz4.frame
import msgpack
import orjson

# Monitoring and metrics (Prometheus)
from prometheus_client import Counter, Histogram, Gauge

# Configuration
from pydantic_settings import BaseSettings, SettingsConfigDict # Correct import for BaseSettings
from pydantic import Field

# Logging setup
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics - Ensure labels are consistent across usage
cache_hits = Counter('cache_hits_total', 'Total cache hits', ['cache_type', 'key_type'])
cache_misses = Counter('cache_misses_total', 'Total cache misses', ['cache_type', 'key_type'])
cache_operations = Counter('cache_operations_total', 'Total cache operations', ['operation', 'cache_type'])
cache_operation_duration = Histogram('cache_operation_duration_seconds', 'Cache operation duration', ['operation', 'cache_type'])
cache_memory_usage = Gauge('cache_memory_usage_bytes', 'Cache memory usage in bytes', ['cache_type'])
cache_size_gauge = Gauge('cache_size_items', 'Number of items in cache', ['cache_type'])

class CacheType(Enum):
    """Types of cache storage"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    GZIP = "gzip"

class SerializationType(Enum):
    """Serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    ORJSON = "orjson"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    compression: CompressionType = CompressionType.NONE
    serialization: SerializationType = SerializationType.PICKLE
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheStats:
    """Cache statistics"""
    total_items: int
    total_size_bytes: int
    hit_rate: float
    miss_rate: float
    avg_access_time: float # Placeholder, typically not universally tracked
    memory_usage: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]

class CacheConfig(BaseSettings):
    """Cache configuration"""
    model_config = SettingsConfigDict(env_prefix='INYANDIKO_CACHE_', extra='ignore')

    # Redis configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: Optional[str] = Field(default=None)
    redis_db: int = Field(default=0)
    redis_max_connections: int = Field(default=20)
    
    # Memory cache configuration
    memory_cache_size: int = Field(default=1000)
    memory_cache_ttl: int = Field(default=3600)  # seconds
    
    # Disk cache configuration
    disk_cache_dir: str = Field(default="./cache")
    disk_cache_size: int = Field(default=1024*1024*1024)  # 1GB
    
    # General configuration
    default_ttl: int = Field(default=3600)
    max_key_length: int = Field(default=250)
    compression_threshold: int = Field(default=1024)  # bytes
    enable_compression: bool = Field(default=True)
    default_compression: CompressionType = Field(default=CompressionType.LZ4)
    default_serialization: SerializationType = Field(default=SerializationType.ORJSON)

class CompressionManager:
    """Handle compression and decompression"""
    
    @staticmethod
    def compress(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm"""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.compress(data)
        elif compression_type == CompressionType.LZ4:
            return lz4.frame.compress(data)
        elif compression_type == CompressionType.GZIP:
            return gzip.compress(data)
        else:
            logger.warning(f"Unknown compression type: {compression_type.value}. Returning uncompressed data.")
            return data
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm"""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression_type == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        elif compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        else:
            logger.warning(f"Unknown decompression type: {compression_type.value}. Returning raw data.")
            return data

class SerializationManager:
    """Handle serialization and deserialization"""
    
    @staticmethod
    def serialize(obj: Any, serialization_type: SerializationType) -> bytes:
        """Serialize object to bytes"""
        try:
            if serialization_type == SerializationType.JSON:
                return orjson.dumps(obj, default=str)
            elif serialization_type == SerializationType.PICKLE:
                return pickle.dumps(obj)
            elif serialization_type == SerializationType.MSGPACK:
                packed = msgpack.packb(obj, default=str, use_bin_type=True)
                if packed is None:
                    raise ValueError("msgpack.packb returned None")
                return packed
            elif serialization_type == SerializationType.ORJSON:
                return orjson.dumps(obj, default=str)
            else:
                logger.warning(f"Unknown serialization type: {serialization_type.value}. Falling back to pickle.")
                return pickle.dumps(obj)
        except Exception as e:
            logger.error(f"Serialization failed for type {serialization_type.value}: {e}", exc_info=True)
            raise

    @staticmethod
    def deserialize(data: bytes, serialization_type: SerializationType) -> Any:
        """Deserialize bytes to object"""
        try:
            if serialization_type == SerializationType.JSON:
                return json.loads(data.decode('utf-8'))
            elif serialization_type == SerializationType.PICKLE:
                return pickle.loads(data)
            elif serialization_type == SerializationType.MSGPACK:
                return msgpack.unpackb(data, raw=False)
            elif serialization_type == SerializationType.ORJSON:
                return orjson.loads(data)
            else:
                logger.warning(f"Unknown deserialization type: {serialization_type.value}. Falling back to pickle.")
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization failed for type {serialization_type.value}: {e}", exc_info=True)
            raise

class MemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used items"""
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats['evictions'] += 1
                logger.debug(f"MemoryCache: Evicted LRU key '{lru_key}'")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.expires_at is None:
            return False
        return datetime.now() > entry.expires_at
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            start_time = time.perf_counter()
            if key not in self._cache:
                self._stats['misses'] += 1
                cache_misses.labels(cache_type='memory', key_type='unknown').inc()
                cache_operation_duration.labels(operation='get', cache_type='memory').observe(time.perf_counter() - start_time)
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if self._is_expired(entry):
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats['misses'] += 1
                cache_misses.labels(cache_type='memory', key_type='unknown').inc()
                cache_operation_duration.labels(operation='get', cache_type='memory').observe(time.perf_counter() - start_time)
                logger.debug(f"MemoryCache: Key '{key}' expired.")
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._update_access_order(key)
            
            self._stats['hits'] += 1
            cache_hits.labels(cache_type='memory', key_type='unknown').inc()
            cache_operation_duration.labels(operation='get', cache_type='memory').observe(time.perf_counter() - start_time)
            logger.debug(f"MemoryCache: Hit for key '{key}'.")
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[Set[str]] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            start_time = time.perf_counter()
            self._evict_lru()
            
            actual_ttl = ttl if ttl is not None else self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=actual_ttl) if actual_ttl > 0 else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=self._estimate_size(value),
                tags=tags or set()
            )
            
            self._cache[key] = entry
            self._update_access_order(key)
            self._stats['sets'] += 1
            cache_operations.labels(operation='set', cache_type='memory').inc()
            cache_operation_duration.labels(operation='set', cache_type='memory').observe(time.perf_counter() - start_time)
            logger.debug(f"MemoryCache: Set key '{key}'. Current size: {len(self._cache)}")
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            start_time = time.perf_counter()
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats['deletes'] += 1
                cache_operations.labels(operation='delete', cache_type='memory').inc()
                cache_operation_duration.labels(operation='delete', cache_type='memory').observe(time.perf_counter() - start_time)
                logger.debug(f"MemoryCache: Deleted key '{key}'.")
                return True
            cache_operation_duration.labels(operation='delete', cache_type='memory').observe(time.perf_counter() - start_time)
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats['hits'] = 0
            self._stats['misses'] = 0
            self._stats['sets'] = 0
            self._stats['deletes'] = 0
            self._stats['evictions'] = 0
            cache_operations.labels(operation='clear', cache_type='memory').inc()
            logger.info("MemoryCache: Cleared all entries.")
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes (using pickle as a proxy)"""
        try:
            return len(pickle.dumps(obj))
        except Exception as e:
            logger.warning(f"MemoryCache: Could not accurately estimate size of object: {e}. Returning default size 100.")
            return 100
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            total_items = len(self._cache)
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = float(self._stats['hits']) / total_requests if total_requests > 0 else 0.0
            miss_rate = 1.0 - hit_rate
            
            entries = list(self._cache.values())
            oldest_entry: Optional[datetime] = min((e.created_at for e in entries), default=None) if entries else None
            newest_entry: Optional[datetime] = max((e.created_at for e in entries), default=None) if entries else None
            
            return CacheStats(
                total_items=total_items,
                total_size_bytes=total_size,
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                avg_access_time=0.0,
                memory_usage=total_size,
                oldest_entry=oldest_entry,
                newest_entry=newest_entry
            )
    
    def close(self) -> None:
        """Cleanup resources for MemoryCache (clear all entries)"""
        self.clear()
        logger.info("MemoryCache: Closed.")


class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, config: CacheConfig, redis_client: Optional[aioredis.Redis] = None):
        self.config = config
        self.redis_client_sync: Optional[redis.Redis] = None
        self.async_redis_client: Optional[aioredis.Redis] = redis_client
        self.compression_manager = CompressionManager()
        self.serialization_manager = SerializationManager()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize Redis connections"""
        if self._is_initialized:
            return

        if self.async_redis_client:
            try:
                await self.async_redis_client.ping()
                self._is_initialized = True
                logger.info("Redis cache initialized with provided client.")
                return
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Provided Redis client failed to connect: {e}")
                self.async_redis_client = None

        try:
            pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                password=self.config.redis_password,
                max_connections=self.config.redis_max_connections,
                socket_timeout=5,
                socket_connect_timeout=5,
                decode_responses=False
            )
            self.async_redis_client = aioredis.Redis(connection_pool=pool)
            await self.async_redis_client.ping()
            self._is_initialized = True
            logger.info(f"Redis cache initialized successfully. Connected to {self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}.")
            
        except (ConnectionError, TimeoutError, OSError) as e:
            self._is_initialized = False
            self.async_redis_client = None
            logger.warning(f"Could not connect to Redis at {self.config.redis_host}:{self.config.redis_port}. "
                           f"Reason: {e}. The application will proceed without Redis caching. "
                           "Please ensure Redis is running and accessible to enable distributed caching.")
        except Exception as e:
            self._is_initialized = False
            self.async_redis_client = None
            logger.error(f"An unexpected error occurred while initializing Redis cache: {e}", exc_info=True)
    
    async def _create_cache_key(self, key: str, prefix: str = "inyandiko") -> str:
        """Create prefixed cache key"""
        if len(key) > self.config.max_key_length:
            key = hashlib.sha256(key.encode()).hexdigest()
        return f"{prefix}:{key}"
    
    def _prepare_value(self, value: Any, compression: CompressionType, serialization: SerializationType) -> bytes:
        """Serialize and compress value for storage"""
        serialized_data = self.serialization_manager.serialize(value, serialization)
        
        if (self.config.enable_compression and compression != CompressionType.NONE and 
            len(serialized_data) >= self.config.compression_threshold):
            compressed_data = self.compression_manager.compress(serialized_data, compression)
        else:
            compressed_data = serialized_data
            compression = CompressionType.NONE
        
        metadata = {
            'compression': compression.value,
            'serialization': serialization.value,
            'original_size': len(serialized_data),
            'compressed_size': len(compressed_data),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata_bytes).to_bytes(4, byteorder='big')
        
        return metadata_length + metadata_bytes + compressed_data
    
    def _extract_value(self, data: bytes) -> Optional[Any]:
        """Extract and decompress value from storage"""
        try:
            metadata_length = int.from_bytes(data[:4], byteorder='big')
            metadata_bytes = data[4:4+metadata_length]
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            compressed_data = data[4+metadata_length:]
            
            compression = CompressionType(metadata['compression'])
            serialization = SerializationType(metadata['serialization'])
            
            decompressed_data = self.compression_manager.decompress(compressed_data, compression)
            value = self.serialization_manager.deserialize(decompressed_data, serialization)
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to extract value from Redis cache data: {e}", exc_info=True)
            return None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self._is_initialized or self.async_redis_client is None:
            logger.debug("RedisCache not initialized or client not available. Skipping get operation.")
            return None
        
        cache_key = await self._create_cache_key(key)
        
        try:
            start_time = time.perf_counter()
            data = await self.async_redis_client.get(cache_key)
            
            if data is None:
                self._stats['misses'] += 1
                cache_misses.labels(cache_type='redis', key_type='unknown').inc()
                cache_operation_duration.labels(operation='get', cache_type='redis').observe(time.perf_counter() - start_time)
                return None
            
            value = self._extract_value(data)
            if value is not None:
                self._stats['hits'] += 1
                cache_hits.labels(cache_type='redis', key_type='unknown').inc()
            else:
                self._stats['misses'] += 1
                cache_misses.labels(cache_type='redis', key_type='unknown').inc()
                logger.warning(f"RedisCache: Failed to extract value for key '{key}'. Treating as miss.")
                
            cache_operation_duration.labels(operation='get', cache_type='redis').observe(time.perf_counter() - start_time)
            return value
                
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection error during get for key '{key}': {e}. Treating as miss.")
            self._stats['errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Redis get error for key '{key}': {e}", exc_info=True)
            self._stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  compression: Optional[CompressionType] = None, 
                  serialization: Optional[SerializationType] = None) -> bool:
        """Set value in Redis cache"""
        if not self._is_initialized or self.async_redis_client is None:
            logger.debug("RedisCache not initialized or client not available. Skipping set operation.")
            return False
        
        cache_key = await self._create_cache_key(key)
        actual_ttl = ttl if ttl is not None else self.config.default_ttl
        actual_compression = compression if compression is not None else self.config.default_compression
        actual_serialization = serialization if serialization is not None else self.config.default_serialization
        
        try:
            start_time = time.perf_counter()
            prepared_value = self._prepare_value(value, actual_compression, actual_serialization)
            
            result: Any
            if actual_ttl > 0:
                result = await self.async_redis_client.setex(cache_key, actual_ttl, prepared_value)
            else:
                result = await self.async_redis_client.set(cache_key, prepared_value)
            
            if result:
                self._stats['sets'] += 1
                cache_operations.labels(operation='set', cache_type='redis').inc()
            
            cache_operation_duration.labels(operation='set', cache_type='redis').observe(time.perf_counter() - start_time)
            return bool(result)
                
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection error during set for key '{key}': {e}.")
            self._stats['errors'] += 1
            return False
        except Exception as e:
            logger.error(f"Redis set error for key '{key}': {e}", exc_info=True)
            self._stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self._is_initialized or self.async_redis_client is None:
            logger.debug("RedisCache not initialized or client not available. Skipping delete operation.")
            return False
        
        cache_key = await self._create_cache_key(key)
        
        try:
            start_time = time.perf_counter()
            result = await self.async_redis_client.delete(cache_key)
            
            if result:
                self._stats['deletes'] += 1
                cache_operations.labels(operation='delete', cache_type='redis').inc()
            
            cache_operation_duration.labels(operation='delete', cache_type='redis').observe(time.perf_counter() - start_time)
            return bool(result)
                
        except Exception as e:
            logger.error(f"Redis delete error for key '{key}': {e}", exc_info=True)
            self._stats['errors'] += 1
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self._is_initialized or self.async_redis_client is None:
            logger.debug("RedisCache not initialized or client not available. Skipping clear_pattern operation.")
            return 0
        
        try:
            start_time = time.perf_counter()
            full_pattern = f"inyandiko:{pattern}*"
            keys_to_delete = [k async for k in self.async_redis_client.scan_iter(match=full_pattern)]
            
            if keys_to_delete:
                deleted = await self.async_redis_client.delete(*keys_to_delete)
                cache_operations.labels(operation='clear_pattern', cache_type='redis').inc()
                cache_operation_duration.labels(operation='clear_pattern', cache_type='redis').observe(time.perf_counter() - start_time)
                return int(deleted)
            
            cache_operation_duration.labels(operation='clear_pattern', cache_type='redis').observe(time.perf_counter() - start_time)
            return 0
            
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}", exc_info=True)
            self._stats['errors'] += 1
            return 0

    async def clear(self) -> None:
        """Clear all keys in the configured Redis database."""
        if not self._is_initialized or self.async_redis_client is None:
            logger.debug("RedisCache not initialized or client not available. Skipping clear operation.")
            return

        try:
            await self.async_redis_client.flushdb()
            logger.info(f"RedisCache: Cleared all entries in DB {self.config.redis_db}.")
            cache_operations.labels(operation='clear', cache_type='redis').inc()
        except Exception as e:
            logger.error(f"Redis clear error: {e}", exc_info=True)
            self._stats['errors'] += 1

    async def get_stats(self) -> CacheStats:
        """Get Redis cache statistics"""
        if not self._is_initialized or self.async_redis_client is None:
            return CacheStats(0, 0, 0.0, 1.0, 0.0, 0, None, None)

        assert self.async_redis_client is not None

        try:
            info_memory = await self.async_redis_client.info('memory')
            info_keyspace = await self.async_redis_client.info('keyspace')
            
            db_info = info_keyspace.get(f'db{self.config.redis_db}', {})
            total_items = int(db_info.get('keys', 0)) if isinstance(db_info, dict) else 0
            
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = float(self._stats['hits']) / total_requests if total_requests > 0 else 0.0
            
            return CacheStats(
                total_items=total_items,
                total_size_bytes=int(info_memory.get('used_memory', 0)),
                hit_rate=hit_rate,
                miss_rate=1.0 - hit_rate,
                avg_access_time=0.0,
                memory_usage=int(info_memory.get('used_memory', 0)),
                oldest_entry=None,
                newest_entry=None
            )
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}", exc_info=True)
            self._stats['errors'] += 1
            return CacheStats(0, 0, 0.0, 1.0, 0.0, 0, None, None)

    async def close(self) -> None:
        """Close the Redis client connection pool."""
        if self.async_redis_client:
            try:
                await self.async_redis_client.close()
                if self.redis_client_sync:
                    self.redis_client_sync.close()
                logger.info("RedisCache: Client connections closed.")
            except Exception as e:
                logger.error(f"Error closing Redis client: {e}", exc_info=True)
        self._is_initialized = False


class DiskCache:
    """Persistent disk-based cache"""
    
    def __init__(self, config: CacheConfig, cache_client: Optional[diskcache.Cache] = None):
        self.config = config
        self.cache_dir = config.disk_cache_dir
        self.cache: Optional[diskcache.Cache] = cache_client
        
    async def initialize(self) -> None:
        """Initialize disk cache"""
        if self.cache is None:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.cache = diskcache.Cache(
                    directory=self.cache_dir,
                    size_limit=self.config.disk_cache_size,
                    eviction_policy='least-recently-used'
                )
                logger.info(f"Disk cache initialized successfully at '{self.cache_dir}'.")
            except Exception as e:
                self.cache = None
                logger.error(f"Failed to initialize disk cache at '{self.cache_dir}': {e}", exc_info=True)
        else:
            logger.info("Disk cache initialized with provided client.")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        if not self.cache:
            logger.debug("DiskCache not initialized. Skipping get operation.")
            return None
        
        try:
            start_time = time.perf_counter()
            data = self.cache.get(key)
            
            if data is None:
                cache_misses.labels(cache_type='disk', key_type='unknown').inc()
                cache_operation_duration.labels(operation='get', cache_type='disk').observe(time.perf_counter() - start_time)
                return None
            
            cache_hits.labels(cache_type='disk', key_type='unknown').inc()
            cache_operation_duration.labels(operation='get', cache_type='disk').observe(time.perf_counter() - start_time)
            return data
                
        except Exception as e:
            logger.error(f"Disk cache get error for key '{key}': {e}", exc_info=True)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache"""
        if not self.cache:
            logger.debug("DiskCache not initialized. Skipping set operation.")
            return False
        
        try:
            start_time = time.perf_counter()
            expire_time: Optional[float] = None
            actual_ttl = ttl if ttl is not None else self.config.default_ttl
            if actual_ttl > 0:
                expire_time = time.time() + actual_ttl
            
            result = self.cache.set(key, value, expire=expire_time)
            
            if result:
                cache_operations.labels(operation='set', cache_type='disk').inc()
            
            cache_operation_duration.labels(operation='set', cache_type='disk').observe(time.perf_counter() - start_time)
            return result
                
        except Exception as e:
            logger.error(f"Disk cache set error for key '{key}': {e}", exc_info=True)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from disk cache"""
        if not self.cache:
            logger.debug("DiskCache not initialized. Skipping delete operation.")
            return False
        
        try:
            start_time = time.perf_counter()
            result = self.cache.delete(key)
            if result:
                cache_operations.labels(operation='delete', cache_type='disk').inc()
            cache_operation_duration.labels(operation='delete', cache_type='disk').observe(time.perf_counter() - start_time)
            return result
        except Exception as e:
            logger.error(f"Disk cache delete error for key '{key}': {e}", exc_info=True)
            return False

    async def clear(self) -> None:
        """Clear all entries from the disk cache."""
        if not self.cache:
            logger.debug("DiskCache not initialized. Skipping clear operation.")
            return
        
        try:
            self.cache.clear()
            logger.info(f"DiskCache: Cleared all entries in '{self.cache_dir}'.")
            cache_operations.labels(operation='clear', cache_type='disk').inc()
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}", exc_info=True)

    async def get_stats(self) -> CacheStats:
        """Get disk cache statistics"""
        if not self.cache:
            return CacheStats(0, 0, 0.0, 1.0, 0.0, 0, None, None)

        try:
            total_items = int(len(self.cache))  # type: ignore[arg-type]
            total_size_bytes = int(self.cache.currsize)  # type: ignore[attr-defined]
            
            current_hits = cache_hits.labels(cache_type='disk', key_type='unknown')._value.get()
            current_misses = cache_misses.labels(cache_type='disk', key_type='unknown')._value.get()
            total_requests = current_hits + current_misses
            hit_rate = current_hits / total_requests if total_requests > 0 else 0.0
            miss_rate = 1.0 - hit_rate

            return CacheStats(
                total_items=total_items,
                total_size_bytes=total_size_bytes,
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                avg_access_time=0.0,
                memory_usage=0,
                oldest_entry=None,
                newest_entry=None
            )
        except Exception as e:
            logger.error(f"Failed to get DiskCache stats: {e}", exc_info=True)
            return CacheStats(0, 0, 0.0, 1.0, 0.0, 0, None, None)

    async def close(self) -> None:
        """Close diskcache resources."""
        if self.cache:
            try:
                self.cache.close()
                logger.info(f"DiskCache: Closed diskcache for directory '{self.cache_dir}'.")
            except Exception as e:
                logger.error(f"Error closing DiskCache: {e}", exc_info=True)

class MultiLayerCache:
    """Multi-layer cache with memory, Redis, and disk storage"""
    
    def __init__(self, config: CacheConfig, 
                 memory_cache_client: Optional[MemoryCache] = None,
                 redis_cache_client: Optional[RedisCache] = None,
                 disk_cache_client: Optional[DiskCache] = None):
        self.config = config
        self.memory_cache = memory_cache_client or MemoryCache(config.memory_cache_size, config.memory_cache_ttl)
        self.redis_cache = redis_cache_client or RedisCache(config)
        self.disk_cache = disk_cache_client or DiskCache(config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.cache_layers: List[Tuple[Union[MemoryCache, RedisCache, DiskCache], CacheType]] = [
            (self.memory_cache, CacheType.MEMORY),
            (self.redis_cache, CacheType.REDIS),
            (self.disk_cache, CacheType.DISK)
        ]
    
    async def initialize(self) -> None:
        """Initialize all cache layers"""
        try:
            await self.redis_cache.initialize()
            await self.disk_cache.initialize()
            logger.info("Multi-layer cache initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize multi-layer cache: {e}", exc_info=True)
            raise

    async def get(self, key: str, populate_upper_layers: bool = True) -> Optional[Any]:
        """Get value from cache hierarchy"""
        value: Optional[Any] = None
        found_layer_type: Optional[CacheType] = None
        
        for cache_layer, layer_type in self.cache_layers:
            try:
                if isinstance(cache_layer, MemoryCache):
                    value = cache_layer.get(key)
                else:
                    value = await cache_layer.get(key)
                
                if value is not None:
                    found_layer_type = layer_type
                    logger.debug(f"MultiLayerCache: Cache hit for key '{key}' in {layer_type.value} layer.")
                    break
                    
            except Exception as e:
                logger.warning(f"Error accessing {layer_type.value} cache for key '{key}': {e}")
                continue
        
        if value is not None and populate_upper_layers and found_layer_type:
            await self._populate_upper_layers(key, value, found_layer_type)
        
        return value
    
    async def _populate_upper_layers(self, key: str, value: Any, found_layer: CacheType) -> None:
        """Populate upper cache layers with found value"""
        for cache_layer, layer_type in reversed(self.cache_layers):
            if layer_type == found_layer:
                break
            
            ttl = self.config.default_ttl // 2
            
            try:
                if isinstance(cache_layer, MemoryCache):
                    cache_layer.set(key, value, ttl=ttl)
                else:
                    await cache_layer.set(key, value, ttl=ttl)
                logger.debug(f"MultiLayerCache: Populated key '{key}' to {layer_type.value} layer.")
            except Exception as e:
                logger.warning(f"Failed to populate upper cache layer {layer_type.value} for key '{key}': {e}")
                    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  layers: Optional[List[CacheType]] = None) -> bool:
        """Set value in specified cache layers"""
        actual_ttl = ttl if ttl is not None else self.config.default_ttl
        target_layers = layers if layers is not None else [CacheType.MEMORY, CacheType.REDIS, CacheType.DISK]
        
        results: List[bool] = []
        
        for cache_layer, layer_type in self.cache_layers:
            if layer_type not in target_layers:
                continue
            
            try:
                if isinstance(cache_layer, MemoryCache):
                    result: bool = cache_layer.set(key, value, ttl=actual_ttl)
                else:
                    result: bool = await cache_layer.set(key, value, ttl=actual_ttl)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to set in {layer_type.value} cache for key '{key}': {e}")
                results.append(False)
        
        return any(results)
    
    async def delete(self, key: str, layers: Optional[List[CacheType]] = None) -> bool:
        """Delete key from specified cache layers"""
        target_layers = layers if layers is not None else [CacheType.MEMORY, CacheType.REDIS, CacheType.DISK]
        
        results: List[bool] = []
        
        for cache_layer, layer_type in self.cache_layers:
            if layer_type not in target_layers:
                continue
            
            try:
                if isinstance(cache_layer, MemoryCache):
                    result: bool = cache_layer.delete(key)
                else:
                    result: bool = await cache_layer.delete(key)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to delete from {layer_type.value} cache for key '{key}': {e}")
                results.append(False)
        
        return any(results)
    
    async def clear_all(self) -> None:
        """Clear all cache layers"""
        for cache_layer, layer_type in self.cache_layers:
            try:
                if hasattr(cache_layer, 'clear'):
                    if isinstance(cache_layer, MemoryCache):
                        cache_layer.clear()
                    else:
                        await cache_layer.clear()
            except Exception as e:
                logger.warning(f"Failed to clear {layer_type.value} cache: {e}")
        logger.info("MultiLayerCache: All cache layers cleared.")
    
    async def get_combined_stats(self) -> Dict[str, CacheStats]:
        """Get statistics from all cache layers"""
        stats: Dict[str, CacheStats] = {}
        
        for cache_layer, layer_type in self.cache_layers:
            try:
                if hasattr(cache_layer, 'get_stats'):
                    if isinstance(cache_layer, MemoryCache):
                        layer_stats: CacheStats = cache_layer.get_stats()
                    else:
                        layer_stats: CacheStats = await cache_layer.get_stats()
                    
                    stats[layer_type.value] = layer_stats
                    
            except Exception as e:
                logger.warning(f"Failed to get stats from {layer_type.value} cache: {e}")
        
        return stats

    async def close(self) -> None:
        """Close all cache layers and associated resources."""
        logger.info("MultiLayerCache: Shutting down all cache layers...")
        for cache_layer, layer_type in self.cache_layers:
            if hasattr(cache_layer, 'close'):
                if isinstance(cache_layer, MemoryCache):
                    cache_layer.close()
                else:
                    await cache_layer.close()
        self.executor.shutdown(wait=True)
        logger.info("MultiLayerCache: All cache layers shut down.")

class CacheManager:
    """Main cache manager with advanced features"""
    
    def __init__(self, config: Optional[CacheConfig] = None,
                 memory_cache_client: Optional[MemoryCache] = None,
                 redis_cache_client: Optional[RedisCache] = None,
                 disk_cache_client: Optional[DiskCache] = None):
        self.config = config or CacheConfig()
        self.cache = MultiLayerCache(self.config, memory_cache_client, redis_cache_client, disk_cache_client)
        self.cache_warming_tasks: Set[asyncio.Task] = set()
        self.background_tasks: Set[asyncio.Task] = set()
        self._running = asyncio.Event()
        self._background_tasks_started = False
        self._shutdown_complete = asyncio.Event()
        
    async def initialize(self) -> None:
        """Initialize cache manager and start background tasks"""
        logger.info("Cache manager: Initializing...")
        await self.cache.initialize()
        
        if not self._background_tasks_started:
            self._running.set()
            self._shutdown_complete.clear()
            
            cleanup_task = asyncio.create_task(self._cleanup_expired_entries(), name="CacheCleanupTask")
            metrics_task = asyncio.create_task(self._update_metrics(), name="CacheMetricsTask")
            
            self.background_tasks.add(cleanup_task)
            self.background_tasks.add(metrics_task)
            
            cleanup_task.add_done_callback(lambda t: self.background_tasks.discard(t) or self._check_background_tasks_completion())
            metrics_task.add_done_callback(lambda t: self.background_tasks.discard(t) or self._check_background_tasks_completion())
            self._background_tasks_started = True
        
        logger.info("Cache manager initialized successfully.")
    
    def _check_background_tasks_completion(self) -> Optional[bool]:
        """Check if all background tasks are done and set shutdown_complete event."""
        # Use `_running.is_set()` to ensure we only consider completion if shutdown was initiated.
        # Check `self.background_tasks` length instead of the set itself for clarity.
        if not self._running.is_set() and len(self.background_tasks) == 0:
            self._shutdown_complete.set()
            return True
        return False

    async def get_or_compute(self, key: str, compute_func: Callable[..., Any], ttl: Optional[int] = None, 
                           force_refresh: bool = False) -> Any:
        """Get value from cache or compute if not found"""
        if not force_refresh:
            cached_value = await self.cache.get(key)
            if cached_value is not None:
                return cached_value
        
        try:
            logger.debug(f"CacheManager: Cache miss for key '{key}', computing value.")
            if asyncio.iscoroutinefunction(compute_func):
                computed_value = await compute_func()
            else:
                computed_value = await asyncio.to_thread(compute_func)
            
            await self.cache.set(key, computed_value, ttl)
            
            return computed_value
            
        except Exception as e:
            logger.error(f"Failed to compute value for key {key}: {e}", exc_info=True)
            if not force_refresh:
                fallback_value = await self.cache.get(key, populate_upper_layers=False)
                if fallback_value is not None:
                    logger.warning(f"Returning stale value for key '{key}' due to computation failure.")
                    return fallback_value
            raise
    
    async def warm_cache(self, warming_config: Dict[str, Dict[str, Any]]) -> None:
        """Warm cache with predefined data"""
        warming_tasks: List[asyncio.Task] = []
        
        for key, config in warming_config.items():
            compute_func = config.get('compute_func')
            ttl = config.get('ttl')
            
            if compute_func:
                task = asyncio.create_task(
                    self.get_or_compute(key, compute_func, ttl, force_refresh=True),
                    name=f"CacheWarmingTask_{key}"
                )
                warming_tasks.append(task)
                self.cache_warming_tasks.add(task)
                task.add_done_callback(self.cache_warming_tasks.discard)
            else:
                logger.warning(f"Cache warming config for key '{key}' missing 'compute_func'. Skipping.")

        if warming_tasks:
            logger.info(f"CacheManager: Waiting for {len(warming_tasks)} cache warming tasks to complete...")
            await asyncio.gather(*warming_tasks, return_exceptions=True)
            logger.info("CacheManager: All cache warming tasks completed.")
        else:
            logger.info("CacheManager: No cache warming tasks to run.")

    async def _cleanup_expired_entries(self) -> None:
        """Background task to cleanup expired entries"""
        while self._running.is_set():
            try:
                logger.debug("CacheManager: Running periodic expired entry cleanup check (no explicit action in this layer).")
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup background task: {e}", exc_info=True)
                await asyncio.sleep(60)
        logger.info("Cache cleanup background task stopped.")

    async def _update_metrics(self) -> None:
        """Background task to update Prometheus metrics"""
        while self._running.is_set():
            try:
                stats = await self.cache.get_combined_stats()
                
                for layer_name, layer_stats in stats.items():
                    cache_size_gauge.labels(cache_type=layer_name).set(layer_stats.total_items)
                    cache_memory_usage.labels(cache_type=layer_name).set(layer_stats.total_size_bytes)
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating cache metrics background task: {e}", exc_info=True)
                await asyncio.sleep(60)
        logger.info("Cache metrics update background task stopped.")
    
    async def invalidate_by_tags(self, tags: Set[str]) -> None:
        """Invalidate cache entries by tags (placeholder for future implementation)"""
        logger.info(f"CacheManager: Invalidating cache entries with tags: {tags} (Not fully implemented in current layers).")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        health = {
            'status': 'healthy',
            'layers': {},
            'errors': []
        }
        
        health['layers']['memory'] = 'healthy'
        
        if self.cache.redis_cache._is_initialized and (redis_client := self.cache.redis_cache.async_redis_client):
            try:
                if await redis_client.ping():
                    health['layers']['redis'] = 'healthy'
                else:
                    health['layers']['redis'] = 'unhealthy'
                    health['errors'].append('Redis ping failed')
            except Exception as e:
                health['layers']['redis'] = 'unhealthy'
                health['errors'].append(f'Redis connection error: {str(e)}')
        else:
            health['layers']['redis'] = 'disconnected'
        
        if disk_client := self.cache.disk_cache.cache:
            try:
                test_key = "health_check_disk_key"
                test_value = "health_check_value"
                await self.cache.disk_cache.set(test_key, test_value, ttl=10)
                retrieved_value = await self.cache.disk_cache.get(test_key)
                if retrieved_value == test_value:
                    health['layers']['disk'] = 'healthy'
                else:
                    health['layers']['disk'] = 'unhealthy'
                    health['errors'].append('Disk cache read/write check failed')
                await self.cache.disk_cache.delete(test_key)
            except Exception as e:
                health['layers']['disk'] = 'unhealthy'
                health['errors'].append(f'Disk cache error: {str(e)}')
        else:
            health['layers']['disk'] = 'not_initialized'
            health['errors'].append('Disk cache not initialized')
        
        if health['errors']:
            health['status'] = 'degraded'
        
        return health

    async def close(self) -> None:
        """Gracefully shut down the CacheManager and all its underlying layers."""
        logger.info("CacheManager: Shutting down...")
        self._running.clear()
        
        if self._background_tasks_started:
            logger.info("CacheManager: Waiting for background tasks to complete...")
            try:
                await asyncio.wait_for(self._shutdown_complete.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("CacheManager: Timeout waiting for background tasks to complete.")
            logger.info("CacheManager: All background tasks stopped.")

        if self.cache_warming_tasks:
            logger.warning(f"CacheManager: Cancelling {len(self.cache_warming_tasks)} outstanding warming tasks.")
            for task in list(self.cache_warming_tasks):
                task.cancel()
            await asyncio.gather(*self.cache_warming_tasks, return_exceptions=True)
            self.cache_warming_tasks.clear()

        await self.cache.close()
        logger.info("CacheManager: Shut down complete.")

    async def get_transcription_cache(self, audio_hash: str) -> Optional[Dict[str, Any]]:
        """Get transcription from cache."""
        return cast(Optional[Dict[str, Any]], await self.cache.get(f"transcription:{audio_hash}"))
    
    async def cache_transcription(self, audio_hash: str, result: Dict[str, Any]):
        """Cache transcription result."""
        await self.cache.set(f"transcription:{audio_hash}", result)
    
    async def get_tts_cache(self, tts_hash: str) -> Optional[bytes]:
        """Get TTS audio from cache."""
        cached = await self.cache.get(f"tts:{tts_hash}")
        return cast(Optional[bytes], cached) if cached else None
    
    async def cache_tts(self, tts_hash: str, audio_bytes: bytes):
        """Cache TTS audio bytes."""
        await self.cache.set(f"tts:{tts_hash}", audio_bytes)
    
    async def health_check(self) -> bool:
        """Check health of all cache layers."""
        health = await self.get_health_status()
        return health['status'] == 'healthy'


# --- Global Cache Manager Instance (for production integration) ---
_global_cache_manager: Optional[CacheManager] = None
_global_cache_manager_lock = asyncio.Lock()

async def get_cache_manager() -> CacheManager:
    """
    Retrieves or initializes the global CacheManager instance.
    This is intended for production use where a single, shared instance is needed.
    """
    global _global_cache_manager
    async with _global_cache_manager_lock:
        if _global_cache_manager is None:
            logger.info("Initializing global CacheManager instance...")
            _global_cache_manager = CacheManager()
            await _global_cache_manager.initialize()
            logger.info("Global CacheManager instance initialized.")
    return _global_cache_manager

# Convenience functions for easy access in the main application
async def cache_get(key: str, populate_upper_layers: bool = True) -> Optional[Any]:
    """Get value from the global cache manager."""
    manager = await get_cache_manager()
    return await manager.cache.get(key, populate_upper_layers=populate_upper_layers)

async def cache_set(key: str, value: Any, ttl: Optional[int] = None, layers: Optional[List[CacheType]] = None) -> bool:
    """Set value in the global cache manager."""
    manager = await get_cache_manager()
    return await manager.cache.set(key, value, ttl=ttl, layers=layers)

async def cache_delete(key: str, layers: Optional[List[CacheType]] = None) -> bool:
    """Delete key from the global cache manager."""
    manager = await get_cache_manager()
    return await manager.cache.delete(key, layers=layers)

async def cache_get_or_compute(key: str, compute_func: Callable[..., Any], ttl: Optional[int] = None, force_refresh: bool = False) -> Any:
    """Get value from the global cache manager or compute if not found."""
    manager = await get_cache_manager()
    return await manager.get_or_compute(key, compute_func, ttl=ttl, force_refresh=force_refresh)