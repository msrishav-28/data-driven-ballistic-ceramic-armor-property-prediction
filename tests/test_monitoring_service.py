"""
Tests for the production monitoring and alerting system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.api.services.monitoring_service import (
    MonitoringService, 
    MetricsCollector, 
    AlertManager, 
    Alert, 
    AlertLevel,
    MetricType
)
from src.api.config.alerting_config import AlertingRule, NotificationChannel
from src.api.services.notification_service import NotificationService


class TestMetricsCollector:
    """Test the metrics collection functionality."""
    
    def test_counter_increment(self):
        """Test counter metric increment."""
        collector = MetricsCollector()
        
        # Test basic increment
        collector.increment_counter("test_counter")
        assert collector.get_counter_value("test_counter") == 1.0
        
        # Test increment with value
        collector.increment_counter("test_counter", 5.0)
        assert collector.get_counter_value("test_counter") == 6.0
        
        # Test with labels
        collector.increment_counter("test_counter", 2.0, {"endpoint": "/api/test"})
        assert collector.get_counter_value("test_counter", {"endpoint": "/api/test"}) == 2.0
    
    def test_gauge_set(self):
        """Test gauge metric setting."""
        collector = MetricsCollector()
        
        # Test basic gauge
        collector.set_gauge("cpu_usage", 75.5)
        assert collector.get_gauge_value("cpu_usage") == 75.5
        
        # Test gauge update
        collector.set_gauge("cpu_usage", 80.0)
        assert collector.get_gauge_value("cpu_usage") == 80.0
        
        # Test with labels
        collector.set_gauge("memory_usage", 60.0, {"component": "ml_models"})
        assert collector.get_gauge_value("memory_usage", {"component": "ml_models"}) == 60.0
    
    def test_histogram_record(self):
        """Test histogram metric recording."""
        collector = MetricsCollector()
        
        # Record some values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_histogram("response_time", value)
        
        stats = collector.get_histogram_stats("response_time")
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["p50"] == 3.0
    
    def test_timer_record(self):
        """Test timer metric recording."""
        collector = MetricsCollector()
        
        # Record some durations
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]  # seconds
        for duration in durations:
            collector.record_timer("api_duration", duration)
        
        stats = collector.get_timer_stats("api_duration")
        assert stats["count"] == 5
        assert stats["min_ms"] == 100.0  # 0.1 * 1000
        assert stats["max_ms"] == 500.0  # 0.5 * 1000
        assert stats["mean_ms"] == 300.0  # 0.3 * 1000


class TestAlertManager:
    """Test the alert management functionality."""
    
    def test_threshold_setup(self):
        """Test threshold configuration."""
        collector = MetricsCollector()
        alert_manager = AlertManager(collector)
        
        # Test default thresholds exist
        assert "cpu_usage_percent" in alert_manager.thresholds
        assert "memory_usage_percent" in alert_manager.thresholds
        
        # Test custom threshold
        alert_manager.add_threshold("custom_metric", 50.0, 90.0, 120)
        assert "custom_metric" in alert_manager.thresholds
        assert alert_manager.thresholds["custom_metric"]["warning"] == 50.0
        assert alert_manager.thresholds["custom_metric"]["critical"] == 90.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_threshold_checking(self, mock_disk, mock_memory, mock_cpu):
        """Test threshold checking and alert generation."""
        # Mock system metrics
        mock_cpu.return_value = 85.0  # Above warning threshold
        mock_memory.return_value = Mock(percent=90.0)  # Above warning threshold
        mock_disk.return_value = Mock(used=800, total=1000)  # 80% usage
        
        collector = MetricsCollector()
        alert_manager = AlertManager(collector)
        
        # Check thresholds
        alerts = alert_manager.check_thresholds()
        
        # Should generate alerts for CPU and memory
        assert len(alerts) >= 2
        alert_metrics = [alert.metric_name for alert in alerts]
        assert "cpu_usage_percent" in alert_metrics
        assert "memory_usage_percent" in alert_metrics
    
    def test_alert_resolution(self):
        """Test alert resolution when metrics return to normal."""
        collector = MetricsCollector()
        alert_manager = AlertManager(collector)
        
        # Create a test alert
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.utcnow(),
            metric_name="test_metric",
            current_value=85.0,
            threshold_value=80.0,
            context={}
        )
        
        # Process the alert
        alert_manager._process_alert(alert)
        assert len(alert_manager.get_active_alerts()) == 1
        
        # Resolve the alert
        alert_manager._resolve_alert(alert)
        assert alert.resolved is True
        assert len(alert_manager.get_active_alerts()) == 0


class TestMonitoringService:
    """Test the main monitoring service."""
    
    @pytest.mark.asyncio
    async def test_monitoring_service_initialization(self):
        """Test monitoring service initialization."""
        service = MonitoringService()
        
        assert service.metrics_collector is not None
        assert service.alert_manager is not None
        assert service.start_time > 0
    
    @pytest.mark.asyncio
    async def test_health_status_generation(self):
        """Test health status generation."""
        service = MonitoringService()
        
        # Set some test metrics
        service.metrics_collector.set_gauge("cpu_usage_percent", 75.0)
        service.metrics_collector.set_gauge("memory_usage_percent", 60.0)
        service.metrics_collector.increment_counter("http_requests_total", 100)
        
        health_status = service.get_health_status()
        
        assert health_status.overall_status in ["healthy", "warning", "critical"]
        assert "system_resources" in health_status.components
        assert "application" in health_status.components
        assert "ml_models" in health_status.components
        assert health_status.metrics["cpu_usage_percent"] == 75.0
        assert health_status.metrics["memory_usage_percent"] == 60.0
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self):
        """Test monitoring service start and stop."""
        service = MonitoringService()
        
        # Start monitoring
        await service.start_monitoring()
        assert service.monitoring_task is not None
        assert not service.monitoring_task.done()
        
        # Stop monitoring
        await service.stop_monitoring()
        assert service.monitoring_task.done()


class TestNotificationService:
    """Test the notification service."""
    
    @pytest.mark.asyncio
    async def test_structured_log_notification(self):
        """Test structured log notification."""
        service = NotificationService()
        
        # Create test alert
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.utcnow(),
            metric_name="test_metric",
            current_value=85.0,
            threshold_value=80.0,
            context={"test": True}
        )
        
        # Create test rule
        rule = AlertingRule(
            name="test_rule",
            metric_name="test_metric",
            threshold_warning=80.0,
            threshold_critical=90.0,
            check_interval_seconds=60,
            notification_channels=[NotificationChannel.STRUCTURED_LOGS],
            description="Test rule"
        )
        
        # Test notification (should not raise exception)
        try:
            await service._send_structured_log_notification(
                alert, 
                service.alerting_config.get_notification_config(NotificationChannel.STRUCTURED_LOGS),
                rule
            )
            success = True
        except Exception:
            success = False
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_webhook_notification_retry(self):
        """Test webhook notification with retry logic."""
        service = NotificationService()
        
        # Mock HTTP client to simulate failures then success
        with patch.object(service.http_client, 'request') as mock_request:
            # First two attempts fail, third succeeds
            mock_request.side_effect = [
                Mock(status_code=500, text="Server Error"),
                Mock(status_code=503, text="Service Unavailable"),
                Mock(status_code=200, text="OK")
            ]
            
            alert = Alert(
                id="test_alert",
                level=AlertLevel.WARNING,
                title="Test Alert",
                message="Test message",
                timestamp=datetime.utcnow(),
                metric_name="test_metric",
                current_value=85.0,
                threshold_value=80.0,
                context={}
            )
            
            rule = AlertingRule(
                name="test_rule",
                metric_name="test_metric",
                threshold_warning=80.0,
                threshold_critical=90.0,
                check_interval_seconds=60,
                notification_channels=[NotificationChannel.WEBHOOK],
                description="Test rule"
            )
            
            # Mock webhook config
            webhook_config = Mock()
            webhook_config.config = {
                "url": "https://example.com/webhook",
                "method": "POST",
                "headers": {},
                "timeout_seconds": 5,
                "retry_attempts": 3
            }
            
            # Should succeed after retries
            try:
                await service._send_webhook_notification(alert, webhook_config, rule)
                success = True
            except Exception:
                success = False
            
            assert success is True
            assert mock_request.call_count == 3
    
    def test_notification_stats(self):
        """Test notification statistics tracking."""
        service = NotificationService()
        
        # Simulate some notifications
        service.notification_stats["total_sent"] = 10
        service.notification_stats["successful"] = 8
        service.notification_stats["failed"] = 2
        service.notification_stats["by_channel"] = {
            "webhook": {"sent": 5, "failed": 1},
            "slack": {"sent": 3, "failed": 1}
        }
        
        stats = service.get_notification_stats()
        
        assert stats["total_notifications"] == 10
        assert stats["successful_notifications"] == 8
        assert stats["failed_notifications"] == 2
        assert stats["success_rate"] == 80.0
        assert "webhook" in stats["by_channel"]
        assert "slack" in stats["by_channel"]


@pytest.mark.asyncio
async def test_integration_monitoring_and_notifications():
    """Integration test for monitoring and notification systems."""
    # Create monitoring service
    monitoring_service = MonitoringService()
    notification_service = NotificationService()
    
    # Set up test metrics that should trigger alerts
    monitoring_service.metrics_collector.set_gauge("cpu_usage_percent", 90.0)  # Above warning
    monitoring_service.metrics_collector.set_gauge("memory_usage_percent", 88.0)  # Above warning
    
    # Check thresholds (should generate alerts)
    alerts = monitoring_service.alert_manager.check_thresholds()
    
    # Should have generated at least CPU and memory alerts
    assert len(alerts) >= 2
    
    # Test that alerts have proper structure
    for alert in alerts:
        assert alert.id is not None
        assert alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
        assert alert.title is not None
        assert alert.message is not None
        assert alert.current_value > alert.threshold_value
    
    # Test health status reflects the alerts
    health_status = monitoring_service.get_health_status()
    assert health_status.overall_status in ["warning", "critical"]
    assert len(health_status.alerts) >= 2


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])