"""
Ticket generation scenarios for SupportOps Arena.
Defines root causes and generates tickets for easy, medium, and hard difficulty levels.
"""

import random
from datetime import datetime, timedelta
from typing import Any
from env.actions import ActionType
from env.state import (
    UserContext, LogEntry, NetworkStatus, VPNStatus, 
    AuthStatus, ServiceHealth, TaskLevel
)


# ─── Easy Task Root Causes ───────────────────────────────────

TASK_EASY_ROOT_CAUSES = [
    {
        "id": "wifi_adapter_disabled",
        "description": "Network adapter disabled by OS update",
        "category": "hardware",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "correct_remediation_alts": [],
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS, ActionType.INSPECT_NETWORK],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_wrong_credentials",
        "description": "Incorrect saved credentials for SSID",
        "category": "auth",
        "correct_remediation": ActionType.RESET_CREDENTIALS,
        "correct_remediation_alts": [],
        "diagnostic_path": [ActionType.CHECK_AUTHENTICATION, ActionType.INSPECT_LOGS],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_dns_corruption",
        "description": "Local DNS cache corruption",
        "category": "dns",
        "correct_remediation": ActionType.FLUSH_DNS,
        "correct_remediation_alts": [ActionType.RECONFIGURE_CLIENT],
        "diagnostic_path": [ActionType.INSPECT_NETWORK, ActionType.INSPECT_LOGS],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_airplane_mode",
        "description": "Airplane mode enabled by accident",
        "category": "hardware",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "correct_remediation_alts": [],
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS],
        "severity": "low",
        "affected_users": 1
    },
    {
        "id": "wifi_dhcp_exhausted",
        "description": "DHCP lease pool exhausted",
        "category": "network",
        "correct_remediation": ActionType.RESTART_SERVICE,
        "correct_remediation_alts": [ActionType.ESCALATE_TICKET],
        "diagnostic_path": [ActionType.INSPECT_NETWORK, ActionType.RUN_DIAGNOSTIC],
        "severity": "medium",
        "affected_users": 15
    },
    {
        "id": "wifi_wrong_ssid",
        "description": "Client connected to wrong SSID (guest vs corporate)",
        "category": "network",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "correct_remediation_alts": [],
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS, ActionType.INSPECT_NETWORK],
        "severity": "low",
        "affected_users": 1
    },
]


# ─── Medium Task Root Causes ──────────────────────────────────

TASK_MEDIUM_ROOT_CAUSES = [
    {
        "id": "expired_password",
        "description": "User password expired, blocking VPN authentication",
        "category": "auth",
        "correct_remediation": ActionType.RESET_CREDENTIALS,
        "correct_remediation_alts": [ActionType.ESCALATE_TICKET],
        "diagnostic_path": [ActionType.CHECK_AUTHENTICATION, ActionType.INSPECT_LOGS],
        "severity": "medium",
        "affected_users": 1
    },
    {
        "id": "vpn_version_mismatch",
        "description": "VPN client version incompatible with gateway",
        "category": "vpn",
        "correct_remediation": ActionType.RECONFIGURE_CLIENT,
        "correct_remediation_alts": [ActionType.ESCALATE_TICKET],
        "diagnostic_path": [ActionType.QUERY_DEVICE_STATUS, ActionType.SEARCH_INTERNAL_KB],
        "severity": "medium",
        "affected_users": 1
    },
    {
        "id": "dns_split_tunnel_conflict",
        "description": "Split-tunnel DNS configuration conflict",
        "category": "dns",
        "correct_remediation": ActionType.FLUSH_DNS,
        "correct_remediation_alts": [ActionType.RECONFIGURE_CLIENT],
        "diagnostic_path": [ActionType.INSPECT_NETWORK, ActionType.RUN_DIAGNOSTIC],
        "severity": "medium",
        "affected_users": 1
    },
    {
        "id": "mfa_token_desync",
        "description": "MFA token out of sync with authentication server",
        "category": "auth",
        "correct_remediation": ActionType.RESET_CREDENTIALS,
        "correct_remediation_alts": [ActionType.ESCALATE_TICKET],
        "diagnostic_path": [ActionType.CHECK_AUTHENTICATION, ActionType.CONTACT_USER],
        "severity": "medium",
        "affected_users": 1
    },
    {
        "id": "firewall_subnet_block",
        "description": "Firewall blocking specific subnet for remote users",
        "category": "network",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [ActionType.INSPECT_NETWORK, ActionType.RUN_DIAGNOSTIC],
        "severity": "high",
        "affected_users": 8
    },
    {
        "id": "certificate_expiry",
        "description": "VPN SSL certificate expired on gateway",
        "category": "vpn",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [ActionType.INSPECT_LOGS, ActionType.SEARCH_INTERNAL_KB],
        "severity": "high",
        "affected_users": 12
    },
]


# ─── Hard Task Root Causes ────────────────────────────────────

TASK_HARD_ROOT_CAUSES = [
    {
        "id": "sso_token_service_degradation",
        "description": "SSO token service degraded in datacenter-east, affecting multiple applications",
        "category": "sso",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [
            ActionType.INSPECT_LOGS,
            ActionType.CHECK_AUTHENTICATION,
            ActionType.RUN_DIAGNOSTIC
        ],
        "severity": "critical",
        "affected_users": 150,
        "multi_ticket": True
    },
    {
        "id": "ldap_replication_lag",
        "description": "LDAP replication lag causing inconsistent auth across regions",
        "category": "auth",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [
            ActionType.CHECK_AUTHENTICATION,
            ActionType.INSPECT_LOGS,
            ActionType.SEARCH_INTERNAL_KB
        ],
        "severity": "critical",
        "affected_users": 200,
        "multi_ticket": True
    },
    {
        "id": "dns_zone_transfer_failure",
        "description": "DNS zone transfer failure causing resolution inconsistencies",
        "category": "dns",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [
            ActionType.INSPECT_NETWORK,
            ActionType.INSPECT_LOGS,
            ActionType.RUN_DIAGNOSTIC
        ],
        "severity": "critical",
        "affected_users": 300,
        "multi_ticket": True
    },
    {
        "id": "proxy_cascading_failure",
        "description": "Web proxy cascading failure affecting multiple services",
        "category": "network",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [
            ActionType.INSPECT_NETWORK,
            ActionType.INSPECT_LOGS,
            ActionType.SEARCH_INTERNAL_KB
        ],
        "severity": "critical",
        "affected_users": 250,
        "multi_ticket": True
    },
    {
        "id": "certificate_chain_break",
        "description": "Intermediate certificate chain break affecting SSL validation",
        "category": "network",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [
            ActionType.INSPECT_LOGS,
            ActionType.RUN_DIAGNOSTIC,
            ActionType.SEARCH_INTERNAL_KB
        ],
        "severity": "critical",
        "affected_users": 180,
        "multi_ticket": True
    },
    {
        "id": "storage_backend_degradation",
        "description": "Shared storage backend degradation affecting file services",
        "category": "hardware",
        "correct_remediation": ActionType.ESCALATE_TICKET,
        "correct_remediation_alts": [],
        "diagnostic_path": [
            ActionType.QUERY_DEVICE_STATUS,
            ActionType.INSPECT_LOGS,
            ActionType.RUN_DIAGNOSTIC
        ],
        "severity": "critical",
        "affected_users": 220,
        "multi_ticket": True
    },
]


class TicketGenerator:
    """
    Generates realistic IT support tickets with configurable difficulty.
    Supports adversarial sampling weights for adaptive difficulty.
    """
    
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._sampling_weights: dict[str, float] = {}
    
    def set_sampling_weights(self, weights: dict[str, float]) -> None:
        """Update sampling weights from adversary."""
        self._sampling_weights = weights
    
    def generate_ticket(
        self,
        task_level: TaskLevel,
        seed: int | None = None
    ) -> dict[str, Any]:
        """
        Generate a ticket scenario for the given difficulty level.
        
        Returns:
            Dictionary containing all ticket components:
            - root_cause: The ground truth root cause info
            - observation_template: Initial observation data
            - misleading_log_index: Index of misleading log (if any)
        """
        if seed is not None:
            self.rng = random.Random(seed)
        
        if task_level == TaskLevel.EASY:
            return self._generate_easy_ticket()
        elif task_level == TaskLevel.MEDIUM:
            return self._generate_medium_ticket()
        elif task_level == TaskLevel.HARD:
            return self._generate_hard_ticket()
        else:
            raise ValueError(f"Invalid task level: {task_level}")
    
    def _generate_easy_ticket(self) -> dict[str, Any]:
        """Generate easy difficulty ticket (Wi-Fi connectivity)."""
        # Apply sampling weights if available
        causes = TASK_EASY_ROOT_CAUSES.copy()
        if self._sampling_weights:
            weights = [self._sampling_weights.get(c["id"], 1.0) for c in causes]
            root_cause = self.rng.choices(causes, weights=weights, k=1)[0]
        else:
            root_cause = self.rng.choice(causes)
        
        # Generate user context
        user_context = self._generate_user_context()
        
        # Generate initial logs (no misleading entries for easy)
        logs = self._generate_initial_logs_easy(root_cause)
        
        # Create observation template
        observation_template = {
            "ticket_summary": "Hi, I can't connect to Wi-Fi and need help urgently.",
            "user_context": user_context,
            "network_status": NetworkStatus.UNKNOWN,
            "vpn_status": VPNStatus.NA,
            "auth_status": AuthStatus.OK,
            "service_health": {
                "dhcp": ServiceHealth.UNKNOWN,
                "dns": ServiceHealth.UNKNOWN,
                "wifi": ServiceHealth.UNKNOWN,
            },
            "system_logs": logs,
        }
        
        return {
            "root_cause": root_cause,
            "observation_template": observation_template,
            "misleading_log_index": None,
        }
    
    def _generate_medium_ticket(self) -> dict[str, Any]:
        """Generate medium difficulty ticket (VPN access)."""
        causes = TASK_MEDIUM_ROOT_CAUSES.copy()
        if self._sampling_weights:
            weights = [self._sampling_weights.get(c["id"], 1.0) for c in causes]
            root_cause = self.rng.choices(causes, weights=weights, k=1)[0]
        else:
            root_cause = self.rng.choice(causes)
        
        user_context = self._generate_user_context()
        user_context.location = "remote"
        
        # Generate logs with 1 misleading entry
        logs, misleading_idx = self._generate_initial_logs_medium(root_cause)
        
        observation_template = {
            "ticket_summary": "I'm working from home and can't access any internal tools through VPN. It was working fine yesterday.",
            "user_context": user_context,
            "network_status": NetworkStatus.UNKNOWN,
            "vpn_status": VPNStatus.NA,
            "auth_status": AuthStatus.OK,
            "service_health": {
                "vpn_client": ServiceHealth.UNKNOWN,
                "vpn_gateway": ServiceHealth.UNKNOWN,
                "dns": ServiceHealth.UNKNOWN,
            },
            "system_logs": logs,
        }
        
        return {
            "root_cause": root_cause,
            "observation_template": observation_template,
            "misleading_log_index": misleading_idx,
        }
    
    def _generate_hard_ticket(self) -> dict[str, Any]:
        """Generate hard difficulty ticket (multi-service failure)."""
        causes = TASK_HARD_ROOT_CAUSES.copy()
        if self._sampling_weights:
            weights = [self._sampling_weights.get(c["id"], 1.0) for c in causes]
            root_cause = self.rng.choices(causes, weights=weights, k=1)[0]
        else:
            root_cause = self.rng.choice(causes)
        
        user_context = self._generate_user_context()
        
        # Generate logs with multiple misleading entries
        logs, misleading_idx = self._generate_initial_logs_hard(root_cause)
        
        observation_template = {
            "ticket_summary": "Multiple users reporting access issues across different systems. Email and collaboration tools are affected.",
            "user_context": user_context,
            "network_status": NetworkStatus.UNKNOWN,
            "vpn_status": VPNStatus.NA,
            "auth_status": AuthStatus.OK,
            "service_health": {
                "mail": ServiceHealth.UNKNOWN,
                "sharepoint": ServiceHealth.UNKNOWN,
                "teams": ServiceHealth.UNKNOWN,
                "sso": ServiceHealth.UNKNOWN,
            },
            "system_logs": logs,
        }
        
        return {
            "root_cause": root_cause,
            "observation_template": observation_template,
            "misleading_log_index": misleading_idx,
        }
    
    def _generate_user_context(self) -> UserContext:
        """Generate randomized user context."""
        departments = ["Engineering", "Sales", "Finance", "HR", "Operations"]
        roles = ["Manager", "Analyst", "Developer", "Coordinator", "Specialist"]
        devices = ["Laptop", "Desktop", "Tablet"]
        os_versions = ["Windows 11", "Windows 10", "macOS 14", "macOS 13"]
        locations = ["onsite", "remote"]
        
        return UserContext(
            department=self.rng.choice(departments),
            role=self.rng.choice(roles),
            device_type=self.rng.choice(devices),
            os_version=self.rng.choice(os_versions),
            location=self.rng.choice(locations),
        )
    
    def _generate_initial_logs_easy(self, root_cause: dict) -> list[LogEntry]:
        """Generate initial logs for easy task (no misleading entries)."""
        now = datetime.utcnow()
        logs = []
        
        # Add generic startup logs
        logs.append(LogEntry(
            timestamp=(now - timedelta(minutes=5)).isoformat(),
            level="INFO",
            service="NetworkManager",
            message="Network service initialized",
            is_misleading=False
        ))
        
        # Add category-relevant log
        category = root_cause["category"]
        if category == "dns":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=3)).isoformat(),
                level="WARN",
                service="DNSResolver",
                message="DNS resolution timeout for internal.company.com",
                is_misleading=False
            ))
        elif category == "auth":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=2)).isoformat(),
                level="ERROR",
                service="WiFiAuth",
                message="Authentication failed: Invalid credentials",
                is_misleading=False
            ))
        elif category == "hardware":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=1)).isoformat(),
                level="WARN",
                service="DeviceManager",
                message="Network adapter state changed",
                is_misleading=False
            ))
        
        logs.append(LogEntry(
            timestamp=now.isoformat(),
            level="ERROR",
            service="NetworkManager",
            message="Connection attempt failed",
            is_misleading=False
        ))
        
        return logs
    
    def _generate_initial_logs_medium(
        self, root_cause: dict
    ) -> tuple[list[LogEntry], int]:
        """Generate logs for medium task with 1 misleading entry."""
        now = datetime.utcnow()
        logs = []
        
        # Add real logs
        logs.append(LogEntry(
            timestamp=(now - timedelta(minutes=10)).isoformat(),
            level="INFO",
            service="VPNClient",
            message="VPN client started",
            is_misleading=False
        ))
        
        category = root_cause["category"]
        if category == "auth":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=7)).isoformat(),
                level="ERROR",
                service="AuthService",
                message=f"Authentication failed: {root_cause['description']}",
                is_misleading=False
            ))
        elif category == "vpn":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=6)).isoformat(),
                level="WARN",
                service="VPNGateway",
                message=f"VPN handshake failed: {root_cause['description']}",
                is_misleading=False
            ))
        
        # Add misleading log (at random position)
        misleading_messages = [
            "Network latency spike detected (250ms avg)",
            "DNS query timeout for vpn.company.com",
            "Firewall rule evaluation took longer than expected",
        ]
        misleading_log = LogEntry(
            timestamp=(now - timedelta(minutes=4)).isoformat(),
            level="WARN",
            service="NetworkMonitor",
            message=self.rng.choice(misleading_messages),
            is_misleading=True
        )
        
        # Insert misleading log at random position (not first or last)
        insert_pos = self.rng.randint(1, len(logs))
        logs.insert(insert_pos, misleading_log)
        
        logs.append(LogEntry(
            timestamp=now.isoformat(),
            level="ERROR",
            service="VPNClient",
            message="Connection failed after 3 retries",
            is_misleading=False
        ))
        
        return logs, insert_pos
    
    def _generate_initial_logs_hard(
        self, root_cause: dict
    ) -> tuple[list[LogEntry], int | None]:
        """Generate logs for hard task with multiple misleading entries."""
        now = datetime.utcnow()
        logs = []
        
        # Add system logs with contradictory evidence
        logs.append(LogEntry(
            timestamp=(now - timedelta(minutes=15)).isoformat(),
            level="INFO",
            service="SystemMonitor",
            message="All services nominal",
            is_misleading=True  # Contradicts the actual failures
        ))
        
        logs.append(LogEntry(
            timestamp=(now - timedelta(minutes=12)).isoformat(),
            level="WARN",
            service="LoadBalancer",
            message="Backend pool health check degraded",
            is_misleading=False
        ))
        
        # Add a misleading authentication success
        logs.append(LogEntry(
            timestamp=(now - timedelta(minutes=10)).isoformat(),
            level="INFO",
            service="AuthService",
            message="User authentication successful (cached token)",
            is_misleading=True
        ))
        
        category = root_cause["category"]
        if category == "sso":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=8)).isoformat(),
                level="ERROR",
                service="SSOTokenService",
                message="Token validation latency exceeding threshold (3000ms)",
                is_misleading=False
            ))
        elif category == "dns":
            logs.append(LogEntry(
                timestamp=(now - timedelta(minutes=7)).isoformat(),
                level="ERROR",
                service="DNSService",
                message="Zone transfer failed from primary nameserver",
                is_misleading=False
            ))
        
        logs.append(LogEntry(
            timestamp=(now - timedelta(minutes=5)).isoformat(),
            level="WARN",
            service="ServiceMesh",
            message="Multiple service endpoints reporting degraded state",
            is_misleading=False
        ))
        
        logs.append(LogEntry(
            timestamp=now.isoformat(),
            level="ERROR",
            service="IncidentManager",
            message=f"Multiple concurrent failures detected: {root_cause['description']}",
            is_misleading=False
        ))
        
        # Return index of first misleading log
        misleading_idx = 0  # First log is misleading
        return logs, misleading_idx
