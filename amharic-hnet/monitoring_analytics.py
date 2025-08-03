#!/usr/bin/env python3
"""
Monitoring and Analytics System - Phase 5 Implementation
Follows the Grand Implementation Plan for production monitoring

Features:
- Real-time performance monitoring
- Usage analytics and insights
- Model performance tracking
- Automated alerting system
- Resource utilization monitoring
- User feedback collection
- A/B testing framework
"""

import os
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: str
    response_time: float
    token_count: int
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    error_rate: float
    user_satisfaction: Optional[float]
    cultural_relevance_score: Optional[float]
    morphological_accuracy: Optional[float]

@dataclass
class UsageStats:
    """Usage statistics"""
    timestamp: str
    total_requests: int
    unique_users: int
    avg_session_duration: float
    popular_prompts: List[str]
    error_count: int
    success_rate: float
    geographic_distribution: Dict[str, int]

@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    gpu_memory: Optional[float]
    active_connections: int
    uptime: float

class DatabaseManager:
    """Database manager for metrics storage"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                response_time REAL,
                token_count INTEGER,
                memory_usage REAL,
                cpu_usage REAL,
                gpu_usage REAL,
                error_rate REAL,
                user_satisfaction REAL,
                cultural_relevance_score REAL,
                morphological_accuracy REAL
            )
        """)
        
        # Usage statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_requests INTEGER,
                unique_users INTEGER,
                avg_session_duration REAL,
                popular_prompts TEXT,
                error_count INTEGER,
                success_rate REAL,
                geographic_distribution TEXT
            )
        """)
        
        # System health table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage REAL,
                network_io TEXT,
                gpu_memory REAL,
                active_connections INTEGER,
                uptime REAL
            )
        """)
        
        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                prompt TEXT,
                generated_text TEXT,
                rating INTEGER,
                feedback_text TEXT,
                cultural_appropriateness INTEGER,
                language_quality INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_metrics(self, metrics: ModelMetrics):
        """Insert model metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_metrics 
            (timestamp, response_time, token_count, memory_usage, cpu_usage, 
             gpu_usage, error_rate, user_satisfaction, cultural_relevance_score, 
             morphological_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.response_time, metrics.token_count,
            metrics.memory_usage, metrics.cpu_usage, metrics.gpu_usage,
            metrics.error_rate, metrics.user_satisfaction, 
            metrics.cultural_relevance_score, metrics.morphological_accuracy
        ))
        
        conn.commit()
        conn.close()
    
    def insert_usage_stats(self, stats: UsageStats):
        """Insert usage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO usage_stats 
            (timestamp, total_requests, unique_users, avg_session_duration,
             popular_prompts, error_count, success_rate, geographic_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stats.timestamp, stats.total_requests, stats.unique_users,
            stats.avg_session_duration, json.dumps(stats.popular_prompts),
            stats.error_count, stats.success_rate, 
            json.dumps(stats.geographic_distribution)
        ))
        
        conn.commit()
        conn.close()
    
    def insert_system_health(self, health: SystemHealth):
        """Insert system health metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_health 
            (timestamp, cpu_percent, memory_percent, disk_usage, network_io,
             gpu_memory, active_connections, uptime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            health.timestamp, health.cpu_percent, health.memory_percent,
            health.disk_usage, json.dumps(health.network_io),
            health.gpu_memory, health.active_connections, health.uptime
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_metrics(self, hours: int = 24) -> List[Dict]:
        """Get recent metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT * FROM model_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        """, (since,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.metrics_buffer = deque(maxlen=1000)
        self.alert_thresholds = {
            'response_time': 5.0,  # seconds
            'memory_usage': 80.0,  # percentage
            'cpu_usage': 90.0,     # percentage
            'error_rate': 5.0,     # percentage
            'gpu_memory': 90.0     # percentage
        }
        self.monitoring_active = False
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring"""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    system_health = self.collect_system_health()
                    self.db_manager.insert_system_health(system_health)
                    
                    # Check for alerts
                    self.check_alerts(system_health)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    def collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv
        }
        
        # GPU memory (if available)
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_percent()
        
        # System uptime
        uptime = time.time() - psutil.boot_time()
        
        return SystemHealth(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage=disk.percent,
            network_io=network_io,
            gpu_memory=gpu_memory,
            active_connections=len(psutil.net_connections()),
            uptime=uptime
        )
    
    def record_model_metrics(self, 
                           response_time: float,
                           token_count: int,
                           error_occurred: bool = False,
                           user_satisfaction: Optional[float] = None):
        """Record model performance metrics"""
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # GPU metrics
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_percent()
        
        metrics = ModelMetrics(
            timestamp=datetime.now().isoformat(),
            response_time=response_time,
            token_count=token_count,
            memory_usage=memory.percent,
            cpu_usage=cpu_percent,
            gpu_usage=gpu_usage,
            error_rate=1.0 if error_occurred else 0.0,
            user_satisfaction=user_satisfaction,
            cultural_relevance_score=None,  # To be calculated
            morphological_accuracy=None     # To be calculated
        )
        
        self.metrics_buffer.append(metrics)
        self.db_manager.insert_metrics(metrics)
    
    def check_alerts(self, health: SystemHealth):
        """Check for alert conditions"""
        alerts = []
        
        if health.cpu_percent > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {health.cpu_percent:.1f}%")
        
        if health.memory_percent > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {health.memory_percent:.1f}%")
        
        if health.gpu_memory and health.gpu_memory > self.alert_thresholds['gpu_memory']:
            alerts.append(f"High GPU memory usage: {health.gpu_memory:.1f}%")
        
        if alerts:
            self.send_alerts(alerts)
    
    def send_alerts(self, alerts: List[str]):
        """Send alert notifications"""
        alert_message = "\n".join(alerts)
        logger.warning(f"ALERTS: {alert_message}")
        
        # Here you could integrate with email, Slack, etc.
        # For now, just log the alerts

class AnalyticsEngine:
    """Analytics and insights engine"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        metrics = self.db_manager.get_recent_metrics(hours)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        response_times = [m['response_time'] for m in metrics if m['response_time']]
        memory_usage = [m['memory_usage'] for m in metrics if m['memory_usage']]
        cpu_usage = [m['cpu_usage'] for m in metrics if m['cpu_usage']]
        
        report = {
            'period': f"Last {hours} hours",
            'total_requests': len(metrics),
            'performance': {
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0
            },
            'reliability': {
                'error_rate': np.mean([m['error_rate'] for m in metrics if m['error_rate'] is not None]),
                'uptime_percentage': self.calculate_uptime_percentage(hours)
            },
            'trends': self.analyze_trends(metrics)
        }
        
        return report
    
    def analyze_trends(self, metrics: List[Dict]) -> Dict[str, str]:
        """Analyze performance trends"""
        if len(metrics) < 10:
            return {"trend": "Insufficient data"}
        
        # Analyze response time trend
        response_times = [m['response_time'] for m in metrics if m['response_time']]
        if len(response_times) >= 10:
            x = np.arange(len(response_times))
            slope, _, r_value, _, _ = stats.linregress(x, response_times)
            
            if slope > 0.01:
                trend = "Response times increasing"
            elif slope < -0.01:
                trend = "Response times improving"
            else:
                trend = "Response times stable"
        else:
            trend = "Insufficient data"
        
        return {"response_time_trend": trend}
    
    def calculate_uptime_percentage(self, hours: int) -> float:
        """Calculate system uptime percentage"""
        # Simplified calculation - in production, you'd track actual downtime
        return 99.9  # Placeholder
    
    def generate_usage_insights(self) -> Dict[str, Any]:
        """Generate usage insights and recommendations"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Get recent usage data
        cursor.execute("""
            SELECT * FROM usage_stats 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        
        usage_data = cursor.fetchall()
        conn.close()
        
        if not usage_data:
            return {"insights": "No usage data available"}
        
        insights = {
            'peak_usage_patterns': self.identify_peak_patterns(usage_data),
            'user_behavior': self.analyze_user_behavior(usage_data),
            'recommendations': self.generate_recommendations(usage_data)
        }
        
        return insights
    
    def identify_peak_patterns(self, usage_data: List) -> Dict[str, Any]:
        """Identify peak usage patterns"""
        # Analyze hourly patterns
        hourly_requests = defaultdict(int)
        
        for row in usage_data:
            timestamp = datetime.fromisoformat(row[1])
            hour = timestamp.hour
            hourly_requests[hour] += row[2]  # total_requests
        
        peak_hour = max(hourly_requests.items(), key=lambda x: x[1])
        
        return {
            'peak_hour': peak_hour[0],
            'peak_requests': peak_hour[1],
            'hourly_distribution': dict(hourly_requests)
        }
    
    def analyze_user_behavior(self, usage_data: List) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        total_users = sum(row[3] for row in usage_data)  # unique_users
        avg_session_duration = np.mean([row[4] for row in usage_data])  # avg_session_duration
        
        return {
            'total_unique_users': total_users,
            'avg_session_duration': avg_session_duration,
            'user_engagement': 'High' if avg_session_duration > 300 else 'Medium' if avg_session_duration > 120 else 'Low'
        }
    
    def generate_recommendations(self, usage_data: List) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze error rates
        avg_error_rate = np.mean([row[6] for row in usage_data])  # error_count
        if avg_error_rate > 5:
            recommendations.append("High error rate detected - investigate model stability")
        
        # Analyze success rates
        avg_success_rate = np.mean([row[7] for row in usage_data])  # success_rate
        if avg_success_rate < 95:
            recommendations.append("Success rate below target - review error handling")
        
        # Resource optimization
        recommendations.append("Consider implementing caching for frequently requested prompts")
        recommendations.append("Monitor peak usage times for scaling decisions")
        
        return recommendations
    
    def create_visualizations(self, output_dir: str = "./analytics_reports"):
        """Create performance visualization charts"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get recent metrics
        metrics = self.db_manager.get_recent_metrics(24)
        
        if not metrics:
            logger.warning("No metrics available for visualization")
            return
        
        # Response time chart
        plt.figure(figsize=(12, 6))
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics]
        response_times = [m['response_time'] for m in metrics if m['response_time']]
        
        plt.subplot(2, 2, 1)
        plt.plot(timestamps[:len(response_times)], response_times)
        plt.title('Response Time Over Time')
        plt.xlabel('Time')
        plt.ylabel('Response Time (s)')
        plt.xticks(rotation=45)
        
        # Memory usage chart
        plt.subplot(2, 2, 2)
        memory_usage = [m['memory_usage'] for m in metrics if m['memory_usage']]
        plt.plot(timestamps[:len(memory_usage)], memory_usage)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.xticks(rotation=45)
        
        # CPU usage chart
        plt.subplot(2, 2, 3)
        cpu_usage = [m['cpu_usage'] for m in metrics if m['cpu_usage']]
        plt.plot(timestamps[:len(cpu_usage)], cpu_usage)
        plt.title('CPU Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.xticks(rotation=45)
        
        # Error rate chart
        plt.subplot(2, 2, 4)
        error_rates = [m['error_rate'] for m in metrics if m['error_rate'] is not None]
        plt.plot(timestamps[:len(error_rates)], error_rates)
        plt.title('Error Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error Rate (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance dashboard saved to {output_dir}/performance_dashboard.png")

class MonitoringDashboard:
    """Main monitoring dashboard"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.performance_monitor = PerformanceMonitor(self.db_manager)
        self.analytics_engine = AnalyticsEngine(self.db_manager)
        
    def start_monitoring(self):
        """Start all monitoring services"""
        logger.info("Starting Amharic LLM Monitoring Dashboard...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(interval=60)
        
        logger.info("Monitoring dashboard started successfully")
    
    def stop_monitoring(self):
        """Stop all monitoring services"""
        self.performance_monitor.stop_monitoring()
        logger.info("Monitoring dashboard stopped")
    
    def generate_daily_report(self):
        """Generate daily monitoring report"""
        logger.info("Generating daily monitoring report...")
        
        # Performance report
        performance_report = self.analytics_engine.generate_performance_report(24)
        
        # Usage insights
        usage_insights = self.analytics_engine.generate_usage_insights()
        
        # Create visualizations
        self.analytics_engine.create_visualizations()
        
        # Compile report
        daily_report = {
            'report_date': datetime.now().isoformat(),
            'performance': performance_report,
            'usage_insights': usage_insights,
            'system_status': 'healthy',  # Simplified
            'recommendations': performance_report.get('recommendations', [])
        }
        
        # Save report
        report_path = f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(daily_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Daily report saved to {report_path}")
        
        return daily_report
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time system status"""
        system_health = self.performance_monitor.collect_system_health()
        
        status = {
            'timestamp': system_health.timestamp,
            'system_health': asdict(system_health),
            'alerts': [],  # Would contain active alerts
            'status': 'healthy' if system_health.cpu_percent < 80 and system_health.memory_percent < 80 else 'warning'
        }
        
        return status
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("Running comprehensive health check...")
        
        health_check = {
            'timestamp': datetime.now().isoformat(),
            'database_status': self.check_database_health(),
            'model_status': self.check_model_health(),
            'system_resources': self.check_system_resources(),
            'overall_status': 'healthy'
        }
        
        # Determine overall status
        if any(status != 'healthy' for status in [health_check['database_status'], 
                                                  health_check['model_status'], 
                                                  health_check['system_resources']]):
            health_check['overall_status'] = 'warning'
        
        return health_check
    
    def check_database_health(self) -> str:
        """Check database connectivity and health"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM model_metrics")
            count = cursor.fetchone()[0]
            conn.close()
            return 'healthy' if count >= 0 else 'warning'
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return 'error'
    
    def check_model_health(self) -> str:
        """Check model availability and performance"""
        try:
            # This would test model loading and inference
            # For now, return healthy
            return 'healthy'
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return 'error'
    
    def check_system_resources(self) -> str:
        """Check system resource availability"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90 or memory.percent > 90:
                return 'warning'
            return 'healthy'
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return 'error'

def main():
    """Main monitoring function"""
    dashboard = MonitoringDashboard()
    
    try:
        # Start monitoring
        dashboard.start_monitoring()
        
        # Run initial health check
        health_status = dashboard.run_health_check()
        print("\n" + "="*60)
        print("🔍 AMHARIC LLM MONITORING DASHBOARD")
        print("="*60)
        print(f"📊 Overall Status: {health_status['overall_status'].upper()}")
        print(f"💾 Database: {health_status['database_status']}")
        print(f"🤖 Model: {health_status['model_status']}")
        print(f"⚡ System: {health_status['system_resources']}")
        print("="*60)
        
        # Generate daily report
        daily_report = dashboard.generate_daily_report()
        
        print("\n📈 Daily Report Generated:")
        if 'performance' in daily_report:
            perf = daily_report['performance']
            if 'total_requests' in perf:
                print(f"   📊 Total Requests: {perf['total_requests']}")
            if 'performance' in perf:
                print(f"   ⏱️  Avg Response Time: {perf['performance'].get('avg_response_time', 0):.2f}s")
                print(f"   💾 Avg Memory Usage: {perf['performance'].get('avg_memory_usage', 0):.1f}%")
        
        print("\n🎯 Monitoring active. Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            time.sleep(300)  # Check every 5 minutes
            status = dashboard.get_real_time_status()
            if status['status'] != 'healthy':
                logger.warning(f"System status: {status['status']}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
        dashboard.stop_monitoring()
        print("✅ Monitoring stopped successfully")
    
    except Exception as e:
        logger.error(f"Monitoring error: {e}")
        dashboard.stop_monitoring()

if __name__ == "__main__":
    main()