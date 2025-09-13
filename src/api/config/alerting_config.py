"""
Alerting configuration for production monitoring.

This module provides configuration for external alerting systems and
notification channels for the Ceramic Armor ML API monitoring system.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.config import get_settings


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    DATADOG = "datadog"
    NEWRELIC = "newrelic"
    STRUCTURED_LOGS = "structured_logs"


@dataclass
class AlertingRule:
    """Configuration for an alerting rule."""
    name: str
    metric_name: str
    threshold_warning: float
    threshold_critical: float
    check_interval_seconds: int
    notification_channels: List[NotificationChannel]
    description: str
    enabled: bool = True
    tags: Dict[str, str] = None


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True


class AlertingConfig:
    """Main alerting configuration class."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.alerting_rules = self._setup_default_rules()
        self.notification_configs = self._setup_notification_configs()
    
    def _setup_default_rules(self) -> List[AlertingRule]:
        """Setup default alerting rules."""
        return [
            # System resource alerts
            AlertingRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                threshold_warning=80.0,
                threshold_critical=95.0,
                check_interval_seconds=60,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK
                ],
                description="CPU usage is high and may impact performance",
                tags={"category": "system", "severity": "performance"}
            ),
            
            AlertingRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                threshold_warning=85.0,
                threshold_critical=95.0,
                check_interval_seconds=60,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK
                ],
                description="Memory usage is high and may cause out-of-memory errors",
                tags={"category": "system", "severity": "critical"}
            ),
            
            AlertingRule(
                name="high_disk_usage",
                metric_name="disk_usage_percent",
                threshold_warning=85.0,
                threshold_critical=95.0,
                check_interval_seconds=300,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK
                ],
                description="Disk usage is high and may cause storage issues",
                tags={"category": "system", "severity": "critical"}
            ),
            
            # Application performance alerts
            AlertingRule(
                name="high_error_rate",
                metric_name="error_rate_percent",
                threshold_warning=5.0,
                threshold_critical=10.0,
                check_interval_seconds=300,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK,
                    NotificationChannel.SLACK
                ],
                description="HTTP error rate is elevated",
                tags={"category": "application", "severity": "critical"}
            ),
            
            AlertingRule(
                name="slow_response_times",
                metric_name="response_time_p95_ms",
                threshold_warning=2000.0,
                threshold_critical=5000.0,
                check_interval_seconds=300,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK
                ],
                description="API response times are slower than expected",
                tags={"category": "application", "severity": "performance"}
            ),
            
            # ML model alerts
            AlertingRule(
                name="high_prediction_error_rate",
                metric_name="prediction_error_rate",
                threshold_warning=2.0,
                threshold_critical=5.0,
                check_interval_seconds=300,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK,
                    NotificationChannel.SLACK
                ],
                description="ML prediction error rate is elevated",
                tags={"category": "ml_models", "severity": "critical"}
            ),
            
            # External API alerts
            AlertingRule(
                name="external_api_errors",
                metric_name="external_api_error_rate",
                threshold_warning=10.0,
                threshold_critical=25.0,
                check_interval_seconds=300,
                notification_channels=[
                    NotificationChannel.STRUCTURED_LOGS,
                    NotificationChannel.WEBHOOK
                ],
                description="External API error rate is elevated",
                tags={"category": "external_apis", "severity": "warning"}
            )
        ]
    
    def _setup_notification_configs(self) -> List[NotificationConfig]:
        """Setup notification channel configurations."""
        configs = []
        
        # Structured logs (always enabled)
        configs.append(NotificationConfig(
            channel=NotificationChannel.STRUCTURED_LOGS,
            config={
                "log_level": "WARNING",
                "include_context": True,
                "include_metrics": True
            },
            enabled=True
        ))
        
        # Webhook notifications (for external systems)
        webhook_url = getattr(self.settings, 'alert_webhook_url', None)
        if webhook_url:
            configs.append(NotificationConfig(
                channel=NotificationChannel.WEBHOOK,
                config={
                    "url": webhook_url,
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {getattr(self.settings, 'alert_webhook_token', '')}"
                    },
                    "timeout_seconds": 10,
                    "retry_attempts": 3
                },
                enabled=True
            ))
        
        # Slack notifications
        slack_webhook = getattr(self.settings, 'slack_webhook_url', None)
        if slack_webhook:
            configs.append(NotificationConfig(
                channel=NotificationChannel.SLACK,
                config={
                    "webhook_url": slack_webhook,
                    "channel": getattr(self.settings, 'slack_channel', '#alerts'),
                    "username": "Ceramic Armor ML API",
                    "icon_emoji": ":warning:",
                    "timeout_seconds": 10
                },
                enabled=True
            ))
        
        # PagerDuty integration
        pagerduty_key = getattr(self.settings, 'pagerduty_integration_key', None)
        if pagerduty_key:
            configs.append(NotificationConfig(
                channel=NotificationChannel.PAGERDUTY,
                config={
                    "integration_key": pagerduty_key,
                    "api_url": "https://events.pagerduty.com/v2/enqueue",
                    "timeout_seconds": 10
                },
                enabled=True
            ))
        
        # Datadog integration
        datadog_api_key = getattr(self.settings, 'datadog_api_key', None)
        if datadog_api_key:
            configs.append(NotificationConfig(
                channel=NotificationChannel.DATADOG,
                config={
                    "api_key": datadog_api_key,
                    "app_key": getattr(self.settings, 'datadog_app_key', ''),
                    "api_url": "https://api.datadoghq.com/api/v1/events",
                    "tags": [
                        f"service:ceramic-armor-ml-api",
                        f"environment:{self.settings.environment}",
                        "team:ml-research"
                    ]
                },
                enabled=True
            ))
        
        return configs
    
    def get_alerting_rules(self) -> List[AlertingRule]:
        """Get all alerting rules."""
        return [rule for rule in self.alerting_rules if rule.enabled]
    
    def get_notification_configs(self) -> List[NotificationConfig]:
        """Get all notification configurations."""
        return [config for config in self.notification_configs if config.enabled]
    
    def get_rule_by_name(self, name: str) -> Optional[AlertingRule]:
        """Get alerting rule by name."""
        for rule in self.alerting_rules:
            if rule.name == name:
                return rule
        return None
    
    def get_notification_config(self, channel: NotificationChannel) -> Optional[NotificationConfig]:
        """Get notification configuration for a specific channel."""
        for config in self.notification_configs:
            if config.channel == channel:
                return config
        return None
    
    def add_custom_rule(self, rule: AlertingRule) -> None:
        """Add a custom alerting rule."""
        self.alerting_rules.append(rule)
        self.logger.info(f"Added custom alerting rule: {rule.name}")
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alerting rule."""
        rule = self.get_rule_by_name(rule_name)
        if rule:
            rule.enabled = False
            self.logger.info(f"Disabled alerting rule: {rule_name}")
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alerting rule."""
        rule = self.get_rule_by_name(rule_name)
        if rule:
            rule.enabled = True
            self.logger.info(f"Enabled alerting rule: {rule_name}")
            return True
        return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the alerting configuration."""
        return {
            "total_rules": len(self.alerting_rules),
            "enabled_rules": len([r for r in self.alerting_rules if r.enabled]),
            "notification_channels": len(self.notification_configs),
            "enabled_channels": len([c for c in self.notification_configs if c.enabled]),
            "rules_by_category": {
                "system": len([r for r in self.alerting_rules if r.tags and r.tags.get("category") == "system"]),
                "application": len([r for r in self.alerting_rules if r.tags and r.tags.get("category") == "application"]),
                "ml_models": len([r for r in self.alerting_rules if r.tags and r.tags.get("category") == "ml_models"]),
                "external_apis": len([r for r in self.alerting_rules if r.tags and r.tags.get("category") == "external_apis"])
            },
            "available_channels": [channel.value for channel in NotificationChannel],
            "configured_channels": [config.channel.value for config in self.notification_configs if config.enabled]
        }


# Global alerting configuration instance
alerting_config = AlertingConfig()