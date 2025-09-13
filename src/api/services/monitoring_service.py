"""
Production monitoring and alerting service for the Ceramic Armor ML API.

This service provides comprehensive monitoring capabilities including:
- Application performance monitoring with structured logging
- Health check endpoints for external monitoring systems
- Error alerting and system status reporting
- Metrics collection and analysis
- Alert threshold management
"""

import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

from src.config import get_settings
from src.api.middleware.logging import structured_logger


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    context: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    description: str


@dataclass
class HealthStatus:
    """System health status."""
    overall_status: str
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    alerts: List[Alert]
    metrics: Dict[str, float]


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.logger = structured_logger
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        key = self._get_metric_key(name, labels)
        self.counters[key] += value
        
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self.counters[key],
            timestamp=datetime.utcnow(),
            labels=labels or {},
            description=f"Counter metric: {name}"
        )
        
        self.metrics[key].append(metric)
        self._log_metric(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        key = self._get_metric_key(name, labels)
        self.gauges[key] = value
        
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            description=f"Gauge metric: {name}"
        )
        
        self.metrics[key].append(metric)
        self._log_metric(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a histogram."""
        key = self._get_metric_key(name, labels)
        self.histograms[key].append(value)
        
        # Keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            description=f"Histogram metric: {name}"
        )
        
        self.metrics[key].append(metric)
        self._log_metric(metric)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timer duration."""
        key = self._get_metric_key(name, labels)
        self.timers[key].append(duration)
        
        metric = Metric(
            name=name,
            type=MetricType.TIMER,
            value=duration,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            description=f"Timer metric: {name}"
        )
        
        self.metrics[key].append(metric)
        self._log_metric(metric)
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get current counter value."""
        key = self._get_metric_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current gauge value."""
        key = self._get_metric_key(name, labels)
        return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._get_metric_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p90": sorted_values[int(count * 0.9)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics."""
        key = self._get_metric_key(name, labels)
        durations = list(self.timers.get(key, []))
        
        if not durations:
            return {}
        
        sorted_durations = sorted(durations)
        count = len(sorted_durations)
        
        return {
            "count": count,
            "min_ms": min(sorted_durations) * 1000,
            "max_ms": max(sorted_durations) * 1000,
            "mean_ms": (sum(sorted_durations) / count) * 1000,
            "p50_ms": sorted_durations[int(count * 0.5)] * 1000,
            "p90_ms": sorted_durations[int(count * 0.9)] * 1000,
            "p95_ms": sorted_durations[int(count * 0.95)] * 1000,
            "p99_ms": sorted_durations[int(count * 0.99)] * 1000
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: self.get_histogram_stats(k.split(":")[0], 
                                                     json.loads(k.split(":", 1)[1]) if ":" in k else {}) 
                          for k in self.histograms.keys()},
            "timers": {k: self.get_timer_stats(k.split(":")[0], 
                                              json.loads(k.split(":", 1)[1]) if ":" in k else {}) 
                      for k in self.timers.keys()}
        }
    
    def _get_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Generate a unique key for the metric."""
        if labels:
            label_str = json.dumps(labels, sort_keys=True)
            return f"{name}:{label_str}"
        return name
    
    def _log_metric(self, metric: Metric) -> None:
        """Log metric to structured logger."""
        self.logger.log_structured(
            "DEBUG", "metric_recorded",
            metric_name=metric.name,
            metric_type=metric.type.value,
            metric_value=metric.value,
            metric_labels=metric.labels,
            timestamp=metric.timestamp.isoformat()
        )


class AlertManager:
    """Manages alerts and thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.logger = structured_logger
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self) -> None:
        """Setup default monitoring thresholds."""
        self.thresholds = {
            "cpu_usage_percent": {
                "warning": 80.0,
                "critical": 95.0,
                "check_interval": 60  # seconds
            },
            "memory_usage_percent": {
                "warning": 85.0,
                "critical": 95.0,
                "check_interval": 60
            },
            "disk_usage_percent": {
                "warning": 85.0,
                "critical": 95.0,
                "check_interval": 300  # 5 minutes
            },
            "error_rate_percent": {
                "warning": 5.0,
                "critical": 10.0,
                "check_interval": 300
            },
            "response_time_p95_ms": {
                "warning": 2000.0,
                "critical": 5000.0,
                "check_interval": 300
            },
            "prediction_error_rate": {
                "warning": 2.0,
                "critical": 5.0,
                "check_interval": 300
            },
            "external_api_error_rate": {
                "warning": 10.0,
                "critical": 25.0,
                "check_interval": 300
            }
        }
    
    def add_threshold(self, metric_name: str, warning: float, critical: float, 
                     check_interval: int = 300) -> None:
        """Add a custom threshold."""
        self.thresholds[metric_name] = {
            "warning": warning,
            "critical": critical,
            "check_interval": check_interval
        }
    
    def check_thresholds(self) -> List[Alert]:
        """Check all thresholds and generate alerts."""
        new_alerts = []
        
        # Check system resource thresholds
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("cpu_usage_percent", cpu_percent)
            alert = self._check_threshold("cpu_usage_percent", cpu_percent, "CPU Usage")
            if alert:
                new_alerts.append(alert)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("memory_usage_percent", memory.percent)
            alert = self._check_threshold("memory_usage_percent", memory.percent, "Memory Usage")
            if alert:
                new_alerts.append(alert)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.set_gauge("disk_usage_percent", disk_percent)
            alert = self._check_threshold("disk_usage_percent", disk_percent, "Disk Usage")
            if alert:
                new_alerts.append(alert)
            
        except Exception as e:
            self.logger.log_error("threshold_check", e, {"operation": "system_resources"})
        
        # Check application metrics thresholds
        try:
            # Error rate
            total_requests = self.metrics_collector.get_counter_value("http_requests_total")
            total_errors = self.metrics_collector.get_counter_value("http_errors_total")
            
            if total_requests > 0:
                error_rate = (total_errors / total_requests) * 100
                self.metrics_collector.set_gauge("error_rate_percent", error_rate)
                alert = self._check_threshold("error_rate_percent", error_rate, "Error Rate")
                if alert:
                    new_alerts.append(alert)
            
            # Response time P95
            response_time_stats = self.metrics_collector.get_timer_stats("http_request_duration")
            if response_time_stats:
                p95_ms = response_time_stats.get("p95_ms", 0)
                alert = self._check_threshold("response_time_p95_ms", p95_ms, "Response Time P95")
                if alert:
                    new_alerts.append(alert)
            
            # Prediction error rate
            total_predictions = self.metrics_collector.get_counter_value("predictions_total")
            prediction_errors = self.metrics_collector.get_counter_value("prediction_errors_total")
            
            if total_predictions > 0:
                prediction_error_rate = (prediction_errors / total_predictions) * 100
                self.metrics_collector.set_gauge("prediction_error_rate", prediction_error_rate)
                alert = self._check_threshold("prediction_error_rate", prediction_error_rate, "Prediction Error Rate")
                if alert:
                    new_alerts.append(alert)
            
        except Exception as e:
            self.logger.log_error("threshold_check", e, {"operation": "application_metrics"})
        
        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        return new_alerts
    
    def _check_threshold(self, metric_name: str, current_value: float, title: str) -> Optional[Alert]:
        """Check if a metric exceeds its threshold."""
        threshold_config = self.thresholds.get(metric_name)
        if not threshold_config:
            return None
        
        warning_threshold = threshold_config["warning"]
        critical_threshold = threshold_config["critical"]
        
        # Determine alert level
        alert_level = None
        threshold_value = None
        
        if current_value >= critical_threshold:
            alert_level = AlertLevel.CRITICAL
            threshold_value = critical_threshold
        elif current_value >= warning_threshold:
            alert_level = AlertLevel.WARNING
            threshold_value = warning_threshold
        
        if alert_level:
            alert_id = f"{metric_name}_{alert_level.value}"
            
            # Check if alert already exists and is not resolved
            existing_alert = self.alerts.get(alert_id)
            if existing_alert and not existing_alert.resolved:
                return None  # Don't create duplicate alerts
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                level=alert_level,
                title=f"{title} {alert_level.value.title()}",
                message=f"{title} is {current_value:.2f}%, exceeding {alert_level.value} threshold of {threshold_value:.2f}%",
                timestamp=datetime.utcnow(),
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                context={
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "threshold_value": threshold_value,
                    "threshold_type": alert_level.value
                }
            )
            
            return alert
        
        else:
            # Check if we need to resolve existing alerts
            for level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                alert_id = f"{metric_name}_{level.value}"
                existing_alert = self.alerts.get(alert_id)
                if existing_alert and not existing_alert.resolved:
                    self._resolve_alert(existing_alert)
        
        return None
    
    def _process_alert(self, alert: Alert) -> None:
        """Process a new alert."""
        self.alerts[alert.id] = alert
        
        # Log the alert
        self.logger.log_structured(
            alert.level.value.upper(), "alert_triggered",
            alert_id=alert.id,
            alert_level=alert.level.value,
            alert_title=alert.title,
            alert_message=alert.message,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            context=alert.context
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.log_error("alert_callback", e, {"alert_id": alert.id})
    
    def _resolve_alert(self, alert: Alert) -> None:
        """Resolve an existing alert."""
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        
        self.logger.log_structured(
            "INFO", "alert_resolved",
            alert_id=alert.id,
            alert_level=alert.level.value,
            alert_title=alert.title,
            resolved_at=alert.resolved_at.isoformat()
        )
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return list(self.alerts.values())


class MonitoringService:
    """Main monitoring service that coordinates metrics collection and alerting."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.logger = structured_logger
        self.settings = get_settings()
        self.start_time = time.time()
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Setup alert callbacks
        self.alert_manager.add_alert_callback(self._log_alert_to_external_system)
    
    async def start_monitoring(self) -> None:
        """Start the monitoring background task."""
        if self.monitoring_task and not self.monitoring_task.done():
            return  # Already running
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.log_structured("INFO", "monitoring_started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring background task."""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.log_structured("INFO", "monitoring_stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Check thresholds and generate alerts
                alerts = self.alert_manager.check_thresholds()
                
                # Log monitoring cycle
                if alerts:
                    self.logger.log_structured(
                        "INFO", "monitoring_cycle_completed",
                        new_alerts_count=len(alerts),
                        active_alerts_count=len(self.alert_manager.get_active_alerts())
                    )
                
                # Wait for next cycle (60 seconds)
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_error("monitoring_loop", e)
                await asyncio.sleep(60)  # Continue monitoring even if there's an error
    
    def _log_alert_to_external_system(self, alert: Alert) -> None:
        """Send alert to external monitoring systems via notification service."""
        try:
            # Import here to avoid circular imports
            from .notification_service import notification_service
            
            # Schedule the notification to be sent asynchronously
            asyncio.create_task(notification_service.send_alert_notification(alert))
            
            self.logger.log_structured(
                "INFO", "alert_external_notification_scheduled",
                alert_id=alert.id,
                alert_level=alert.level.value,
                alert_title=alert.title,
                notification_scheduled=True
            )
            
        except Exception as e:
            self.logger.log_structured(
                "ERROR", "alert_external_notification_failed",
                alert_id=alert.id,
                alert_level=alert.level.value,
                error=str(e)
            )
    
    def get_health_status(self) -> HealthStatus:
        """Get comprehensive health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall status
        overall_status = "healthy"
        if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
            overall_status = "critical"
        elif any(alert.level == AlertLevel.ERROR for alert in active_alerts):
            overall_status = "error"
        elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
            overall_status = "warning"
        
        # Get component statuses
        components = {
            "system_resources": self._get_system_resources_status(),
            "application": self._get_application_status(),
            "ml_models": self._get_ml_models_status(),
            "external_apis": self._get_external_apis_status()
        }
        
        # Get key metrics
        key_metrics = {
            "uptime_seconds": time.time() - self.start_time,
            "cpu_usage_percent": self.metrics_collector.get_gauge_value("cpu_usage_percent") or 0,
            "memory_usage_percent": self.metrics_collector.get_gauge_value("memory_usage_percent") or 0,
            "error_rate_percent": self.metrics_collector.get_gauge_value("error_rate_percent") or 0,
            "total_requests": self.metrics_collector.get_counter_value("http_requests_total"),
            "total_predictions": self.metrics_collector.get_counter_value("predictions_total")
        }
        
        return HealthStatus(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            components=components,
            alerts=active_alerts,
            metrics=key_metrics
        )
    
    def _get_system_resources_status(self) -> Dict[str, Any]:
        """Get system resources status."""
        try:
            cpu_percent = self.metrics_collector.get_gauge_value("cpu_usage_percent") or 0
            memory_percent = self.metrics_collector.get_gauge_value("memory_usage_percent") or 0
            disk_percent = self.metrics_collector.get_gauge_value("disk_usage_percent") or 0
            
            status = "healthy"
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = "critical"
            elif cpu_percent > 80 or memory_percent > 85 or disk_percent > 85:
                status = "warning"
            
            return {
                "status": status,
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory_percent,
                "disk_usage_percent": disk_percent
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_application_status(self) -> Dict[str, Any]:
        """Get application status."""
        try:
            error_rate = self.metrics_collector.get_gauge_value("error_rate_percent") or 0
            total_requests = self.metrics_collector.get_counter_value("http_requests_total")
            
            status = "healthy"
            if error_rate > 10:
                status = "critical"
            elif error_rate > 5:
                status = "warning"
            
            return {
                "status": status,
                "error_rate_percent": error_rate,
                "total_requests": total_requests,
                "uptime_seconds": time.time() - self.start_time
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_ml_models_status(self) -> Dict[str, Any]:
        """Get ML models status."""
        try:
            prediction_error_rate = self.metrics_collector.get_gauge_value("prediction_error_rate") or 0
            total_predictions = self.metrics_collector.get_counter_value("predictions_total")
            
            status = "healthy"
            if prediction_error_rate > 5:
                status = "critical"
            elif prediction_error_rate > 2:
                status = "warning"
            
            return {
                "status": status,
                "prediction_error_rate": prediction_error_rate,
                "total_predictions": total_predictions
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_external_apis_status(self) -> Dict[str, Any]:
        """Get external APIs status."""
        try:
            api_error_rate = self.metrics_collector.get_gauge_value("external_api_error_rate") or 0
            
            status = "healthy"
            if api_error_rate > 25:
                status = "critical"
            elif api_error_rate > 10:
                status = "warning"
            
            return {
                "status": status,
                "error_rate_percent": api_error_rate
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "metrics": self.metrics_collector.get_all_metrics(),
            "active_alerts_count": len(self.alert_manager.get_active_alerts()),
            "total_alerts_count": len(self.alert_manager.get_all_alerts())
        }


# Global monitoring service instance
monitoring_service = MonitoringService()