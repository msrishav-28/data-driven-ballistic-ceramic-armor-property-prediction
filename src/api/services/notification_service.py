"""
Notification service for sending alerts to external systems.

This service handles sending notifications through various channels
including webhooks, Slack, PagerDuty, and other monitoring systems.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

import httpx
from fastapi import HTTPException

from src.api.config.alerting_config import (
    alerting_config, 
    NotificationChannel, 
    NotificationConfig,
    AlertingRule
)
from src.api.services.monitoring_service import Alert, AlertLevel
from src.api.middleware.logging import structured_logger


class NotificationService:
    """Service for sending notifications to external systems."""
    
    def __init__(self):
        self.logger = structured_logger
        self.alerting_config = alerting_config
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.notification_stats = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0,
            "by_channel": {}
        }
    
    async def send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification through configured channels."""
        # Get alerting rule for this alert
        rule = self.alerting_config.get_rule_by_name(alert.metric_name)
        if not rule:
            self.logger.log_structured(
                "WARNING", "alert_notification_no_rule",
                alert_id=alert.id,
                metric_name=alert.metric_name
            )
            return
        
        # Send notifications through configured channels
        for channel in rule.notification_channels:
            try:
                await self._send_notification(alert, channel, rule)
                self.notification_stats["successful"] += 1
                
                # Update channel stats
                channel_name = channel.value
                if channel_name not in self.notification_stats["by_channel"]:
                    self.notification_stats["by_channel"][channel_name] = {"sent": 0, "failed": 0}
                self.notification_stats["by_channel"][channel_name]["sent"] += 1
                
            except Exception as e:
                self.logger.log_error(
                    "notification_send_failed", e, 
                    {
                        "alert_id": alert.id,
                        "channel": channel.value,
                        "rule_name": rule.name
                    }
                )
                self.notification_stats["failed"] += 1
                
                # Update channel stats
                channel_name = channel.value
                if channel_name not in self.notification_stats["by_channel"]:
                    self.notification_stats["by_channel"][channel_name] = {"sent": 0, "failed": 0}
                self.notification_stats["by_channel"][channel_name]["failed"] += 1
        
        self.notification_stats["total_sent"] += 1
    
    async def _send_notification(self, alert: Alert, channel: NotificationChannel, rule: AlertingRule) -> None:
        """Send notification through a specific channel."""
        config = self.alerting_config.get_notification_config(channel)
        if not config or not config.enabled:
            return
        
        if channel == NotificationChannel.STRUCTURED_LOGS:
            await self._send_structured_log_notification(alert, config, rule)
        elif channel == NotificationChannel.WEBHOOK:
            await self._send_webhook_notification(alert, config, rule)
        elif channel == NotificationChannel.SLACK:
            await self._send_slack_notification(alert, config, rule)
        elif channel == NotificationChannel.PAGERDUTY:
            await self._send_pagerduty_notification(alert, config, rule)
        elif channel == NotificationChannel.DATADOG:
            await self._send_datadog_notification(alert, config, rule)
        else:
            self.logger.log_structured(
                "WARNING", "unsupported_notification_channel",
                channel=channel.value,
                alert_id=alert.id
            )
    
    async def _send_structured_log_notification(self, alert: Alert, config: NotificationConfig, rule: AlertingRule) -> None:
        """Send notification via structured logging."""
        log_level = config.config.get("log_level", "WARNING")
        
        log_data = {
            "alert_id": alert.id,
            "alert_level": alert.level.value,
            "alert_title": alert.title,
            "alert_message": alert.message,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "rule_name": rule.name,
            "notification_channel": "structured_logs"
        }
        
        if config.config.get("include_context", True):
            log_data["context"] = alert.context
        
        if config.config.get("include_metrics", True):
            log_data["timestamp"] = alert.timestamp.isoformat()
        
        self.logger.log_structured(log_level, "alert_notification", **log_data)
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig, rule: AlertingRule) -> None:
        """Send notification via webhook."""
        webhook_config = config.config
        url = webhook_config["url"]
        
        payload = {
            "alert": {
                "id": alert.id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "context": alert.context
            },
            "rule": {
                "name": rule.name,
                "description": rule.description,
                "tags": rule.tags or {}
            },
            "service": {
                "name": "ceramic-armor-ml-api",
                "environment": self.alerting_config.settings.environment
            },
            "notification": {
                "channel": "webhook",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        headers = webhook_config.get("headers", {})
        timeout = webhook_config.get("timeout_seconds", 10)
        retry_attempts = webhook_config.get("retry_attempts", 3)
        
        for attempt in range(retry_attempts):
            try:
                response = await self.http_client.request(
                    method=webhook_config.get("method", "POST"),
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                
                if response.status_code < 400:
                    self.logger.log_structured(
                        "INFO", "webhook_notification_sent",
                        alert_id=alert.id,
                        webhook_url=url,
                        status_code=response.status_code,
                        attempt=attempt + 1
                    )
                    return
                else:
                    self.logger.log_structured(
                        "WARNING", "webhook_notification_failed",
                        alert_id=alert.id,
                        webhook_url=url,
                        status_code=response.status_code,
                        attempt=attempt + 1,
                        response_text=response.text[:500]
                    )
                    
            except Exception as e:
                self.logger.log_structured(
                    "ERROR", "webhook_notification_error",
                    alert_id=alert.id,
                    webhook_url=url,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to send webhook notification after {retry_attempts} attempts")
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig, rule: AlertingRule) -> None:
        """Send notification via Slack webhook."""
        slack_config = config.config
        webhook_url = slack_config["webhook_url"]
        
        # Determine color based on alert level
        color_map = {
            AlertLevel.INFO: "#36a64f",      # Green
            AlertLevel.WARNING: "#ff9500",   # Orange  
            AlertLevel.ERROR: "#ff0000",     # Red
            AlertLevel.CRITICAL: "#8B0000"   # Dark Red
        }
        
        color = color_map.get(alert.level, "#ff9500")
        
        # Create Slack message
        payload = {
            "channel": slack_config.get("channel", "#alerts"),
            "username": slack_config.get("username", "Ceramic Armor ML API"),
            "icon_emoji": slack_config.get("icon_emoji", ":warning:"),
            "attachments": [
                {
                    "color": color,
                    "title": f"🚨 {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{alert.current_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Environment",
                            "value": self.alerting_config.settings.environment,
                            "short": True
                        }
                    ],
                    "footer": "Ceramic Armor ML API Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        timeout = slack_config.get("timeout_seconds", 10)
        
        try:
            response = await self.http_client.post(
                webhook_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                self.logger.log_structured(
                    "INFO", "slack_notification_sent",
                    alert_id=alert.id,
                    channel=slack_config.get("channel", "#alerts")
                )
            else:
                raise Exception(f"Slack API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.log_structured(
                "ERROR", "slack_notification_error",
                alert_id=alert.id,
                error=str(e)
            )
            raise
    
    async def _send_pagerduty_notification(self, alert: Alert, config: NotificationConfig, rule: AlertingRule) -> None:
        """Send notification via PagerDuty."""
        pagerduty_config = config.config
        integration_key = pagerduty_config["integration_key"]
        api_url = pagerduty_config["api_url"]
        
        # Map alert levels to PagerDuty severity
        severity_map = {
            AlertLevel.INFO: "info",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "error", 
            AlertLevel.CRITICAL: "critical"
        }
        
        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "dedup_key": f"ceramic-armor-ml-{alert.metric_name}",
            "payload": {
                "summary": alert.title,
                "source": "ceramic-armor-ml-api",
                "severity": severity_map.get(alert.level, "warning"),
                "component": alert.metric_name,
                "group": rule.tags.get("category", "unknown") if rule.tags else "unknown",
                "class": "monitoring_alert",
                "custom_details": {
                    "alert_id": alert.id,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "environment": self.alerting_config.settings.environment,
                    "rule_name": rule.name,
                    "context": alert.context
                }
            }
        }
        
        timeout = pagerduty_config.get("timeout_seconds", 10)
        
        try:
            response = await self.http_client.post(
                api_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 202:
                self.logger.log_structured(
                    "INFO", "pagerduty_notification_sent",
                    alert_id=alert.id,
                    dedup_key=payload["dedup_key"]
                )
            else:
                raise Exception(f"PagerDuty API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.log_structured(
                "ERROR", "pagerduty_notification_error",
                alert_id=alert.id,
                error=str(e)
            )
            raise
    
    async def _send_datadog_notification(self, alert: Alert, config: NotificationConfig, rule: AlertingRule) -> None:
        """Send notification via Datadog Events API."""
        datadog_config = config.config
        api_key = datadog_config["api_key"]
        api_url = datadog_config["api_url"]
        
        # Map alert levels to Datadog priority
        priority_map = {
            AlertLevel.INFO: "low",
            AlertLevel.WARNING: "normal",
            AlertLevel.ERROR: "high",
            AlertLevel.CRITICAL: "high"
        }
        
        # Map alert levels to Datadog alert type
        alert_type_map = {
            AlertLevel.INFO: "info",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "error",
            AlertLevel.CRITICAL: "error"
        }
        
        payload = {
            "title": alert.title,
            "text": f"{alert.message}\n\nMetric: {alert.metric_name}\nCurrent Value: {alert.current_value:.2f}\nThreshold: {alert.threshold_value:.2f}",
            "priority": priority_map.get(alert.level, "normal"),
            "alert_type": alert_type_map.get(alert.level, "warning"),
            "source_type_name": "ceramic-armor-ml-api",
            "tags": datadog_config.get("tags", []) + [
                f"alert_id:{alert.id}",
                f"metric:{alert.metric_name}",
                f"level:{alert.level.value}",
                f"rule:{rule.name}"
            ]
        }
        
        headers = {
            "DD-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        if datadog_config.get("app_key"):
            headers["DD-APPLICATION-KEY"] = datadog_config["app_key"]
        
        try:
            response = await self.http_client.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code in [200, 202]:
                self.logger.log_structured(
                    "INFO", "datadog_notification_sent",
                    alert_id=alert.id,
                    event_id=response.json().get("event", {}).get("id")
                )
            else:
                raise Exception(f"Datadog API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.log_structured(
                "ERROR", "datadog_notification_error",
                alert_id=alert.id,
                error=str(e)
            )
            raise
    
    async def send_test_notification(self, channel: NotificationChannel) -> Dict[str, Any]:
        """Send a test notification to verify configuration."""
        # Create a test alert
        test_alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            title="Test Alert - Ceramic Armor ML API",
            message="This is a test alert to verify notification configuration.",
            timestamp=datetime.utcnow(),
            metric_name="test_metric",
            current_value=75.0,
            threshold_value=70.0,
            context={"test": True, "environment": self.alerting_config.settings.environment}
        )
        
        # Create a test rule
        test_rule = AlertingRule(
            name="test_rule",
            metric_name="test_metric",
            threshold_warning=70.0,
            threshold_critical=90.0,
            check_interval_seconds=60,
            notification_channels=[channel],
            description="Test rule for notification verification",
            tags={"category": "test", "severity": "test"}
        )
        
        try:
            await self._send_notification(test_alert, channel, test_rule)
            return {
                "success": True,
                "message": f"Test notification sent successfully via {channel.value}",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send test notification via {channel.value}: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            "total_notifications": self.notification_stats["total_sent"],
            "successful_notifications": self.notification_stats["successful"],
            "failed_notifications": self.notification_stats["failed"],
            "success_rate": (
                self.notification_stats["successful"] / max(self.notification_stats["total_sent"], 1)
            ) * 100,
            "by_channel": self.notification_stats["by_channel"],
            "configured_channels": [
                config.channel.value for config in self.alerting_config.get_notification_configs()
            ]
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.aclose()


# Global notification service instance
notification_service = NotificationService()